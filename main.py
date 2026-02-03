from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, List
import pandas as pd
import uuid
import os
from pathlib import Path
from typing import Optional as TypingOptional

from rq import Retry

from services.data_loader import DataLoader
from services.cleaner import DataCleaner
from services.eda import EDAService
from services.outliers import OutlierService
from services.feature_engineering import FeatureEngineeringService
from services.queue import get_queue
import structlog
from services.schema import (
    DatasetResponse,
    DatasetItem,
    GraphInfo,
    GraphsResponse,
    FeaturesResponse,
    CollaborationGraphInfo,
    CollaborationGraphsResponse,
    CollaborationDatasetItem,
    DatasetStatusResponse,
    CleaningReportResponse,
)
from s3_client import upload_file_and_get_key, generate_presigned_url, delete_file
from database import (
    connect_db,
    disconnect_db,
    create_dataset,
    update_dataset,
    get_dataset,
    create_cleaning_log,
    create_outlier_log,
    create_feature_log,
    create_graph_metadata,
    get_cleaning_logs,
    get_outlier_logs,
    get_feature_logs,
    get_graphs_metadata,
    get_graph_metadata_by_id,
    list_datasets,
    create_cleaning_report,
    get_cleaning_report,
    count_user_mode_usage,
    create_user_mode_usage,
)
from middleware.auth import get_current_user, JWTPayload, AuthenticationError
from middleware.rate_limit import (
    rate_limit_dependency,
    add_rate_limit_headers,
    RateLimitInfo,
)
logger = structlog.get_logger()

app = FastAPI(title="NeuraLearn microservices api", version="1.0.0")

# CORS Configuration - restrict origins in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add rate limit headers to responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add rate limit headers if available
        if hasattr(request.state, 'rate_limit_info'):
            info: RateLimitInfo = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info.limit)
            response.headers["X-RateLimit-Remaining"] = str(info.remaining)
            response.headers["X-RateLimit-Reset"] = str(info.reset)
        
        return response


app.add_middleware(RateLimitHeadersMiddleware)

data_loader = DataLoader()
eda_service = EDAService()

STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup():
    await connect_db()


@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()


async def _load_dataset_dataframe(dataset_id: str) -> pd.DataFrame:
    """
    Load the raw dataset for a given dataset_id from storage or URL.

    This is used by both the in-process background task and the Redis/RQ worker job.
    """
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found while processing")

    df: TypingOptional[pd.DataFrame] = None

    # Prefer loading from URL if available
    if getattr(dataset, "rawUrl", None):
        try:
            df = await data_loader.load_from_url(dataset.rawUrl)
        except Exception:
            df = None

    # Fallback to local raw data path or S3 key
    raw_path = getattr(dataset, "rawDataPath", None)
    if df is None and raw_path:
        if os.path.exists(raw_path):
             df = data_loader.load_saved_dataset(raw_path)
        else:
            # Assume it is an S3 key
            s3_url = generate_presigned_url(raw_path)
            if s3_url:
                try:
                    df = await data_loader.load_from_url(s3_url)
                except Exception:
                    df = None

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Raw dataset could not be loaded for processing",
        )

    return df


async def process_dataset(dataset_id: str, mode: str = "fast"):
    log = logger.bind(dataset_id=dataset_id, mode=mode)
    files_to_clean = []
    try:
        log.info("processing_started")
        await update_dataset(dataset_id, status="processing")
        dataset = await get_dataset(dataset_id)
        if not dataset:
            log.warning("dataset_not_found_during_processing")
            return

        df = await _load_dataset_dataframe(dataset_id)
        log.info("dataset_loaded", rows=df.shape[0], columns=df.shape[1])

        cleaner = DataCleaner(mode=mode)
        df_cleaned = cleaner.handle_missing_values(df)
        log.info("cleaning_completed", logs_count=len(cleaner.get_cleaning_logs()))
        
        for cleaning_log in cleaner.get_cleaning_logs():
            await create_cleaning_log(
                dataset_id=dataset_id,
                column=cleaning_log["column"],
                null_count=cleaning_log["null_count"],
                action=cleaning_log["action"],
                reason=cleaning_log["reason"]
            )
        
        eda_results = eda_service.generate_all_eda(df_cleaned, dataset_id)
        log.info("eda_generated", graphs_count=len(eda_results["graphs"]))

        for graph in eda_results["graphs"]:
            local_path = graph["file_path"]
            files_to_clean.append(local_path)
            s3_key = f"graphs/{dataset_id}/{os.path.basename(local_path)}"
            stored_key = upload_file_and_get_key(local_path, s3_key, delete_local=False)
            
            if not stored_key:
                 raise Exception(f"Failed to upload graph to S3: {local_path}")

            await create_graph_metadata(
                dataset_id=dataset_id,
                graph_type=graph["type"],
                column=graph.get("column"),
                file_path=stored_key,
            )
        
        outlier_service = OutlierService()
        df_cleaned = outlier_service.detect_and_fix_outliers(
            df_cleaned, 
            method="IQR", 
            fix_strategy="cap"
        )
        log.info("outliers_processed", logs_count=len(outlier_service.get_outlier_logs()))
        
        for outlier_log in outlier_service.get_outlier_logs():
            await create_outlier_log(
                dataset_id=dataset_id,
                column=outlier_log["column"],
                outlier_count=outlier_log["outlier_count"],
                method=outlier_log["method"],
                action=outlier_log["action"]
            )
        
        feature_service = FeatureEngineeringService()
        df_final = feature_service.apply_feature_engineering(df_cleaned)
        log.info("feature_engineering_completed", logs_count=len(feature_service.get_feature_logs()))
        
        for feature_log in feature_service.get_feature_logs():
            await create_feature_log(
                dataset_id=dataset_id,
                action=feature_log["action"],
                details=feature_log["details"]
            )
        
        if mode == "deep" and cleaner.gemini:
            try:
                log.info("generating_deep_report")
                sample_df = cleaner.get_sample_dataframe()
                if sample_df is not None and len(cleaner.get_cleaning_logs()) > 0:
                    report = cleaner.gemini.get_deep_cleaning_report(
                        df_sample=sample_df,
                        cleaning_logs=cleaner.get_cleaning_logs(),
                        dataset_name=dataset.databaseName or "Dataset"
                    )
                    if report and report.get("reasoning"):
                        recommendations = report.get("recommendations")
                        if isinstance(recommendations, list):
                            recommendations = "\n".join(f"- {rec}" if isinstance(rec, str) else str(rec) for rec in recommendations)
                        elif recommendations is None:
                            recommendations = None
                        elif not isinstance(recommendations, str):
                            recommendations = str(recommendations)
                        
                        await create_cleaning_report(
                            dataset_id=dataset_id,
                            reasoning=report["reasoning"],
                            summary=report["summary"],
                            recommendations=recommendations
                        )
                        log.info("deep_report_generated")
            except Exception as e:
                error_str = str(e)
                log.error("deep_report_generation_failed", error=error_str)
                if "quota" in error_str.lower() or "429" in error_str or "rate limit" in error_str.lower():
                    print(f"Gemini quota exhausted, skipping report generation: {e}")
                else:
                    print(f"Failed to generate deep cleaning report: {e}")
                    await create_cleaning_report(
                        dataset_id=dataset_id,
                        reasoning=f"Dataset cleaned using {len(cleaner.get_cleaning_logs())} actions. AI-assisted cleaning was attempted but report generation encountered an error: {error_str[:100]}",
                        summary=f"Cleaned {len(cleaner.get_cleaning_logs())} columns with missing values. Data quality improved.",
                        recommendations="Review cleaned data for domain-specific validation. Consider retrying for detailed AI-generated report."
                    )
        
        cleaned_path = data_loader.save_cleaned_dataset(df_final, dataset_id)
        files_to_clean.append(cleaned_path)

        cleaned_s3_key = f"datasets/{dataset_id}/cleaned.csv"
        # Using delete_local=False so we can handle cleanup in finally
        cleaned_key = upload_file_and_get_key(cleaned_path, cleaned_s3_key, delete_local=False)
        
        if not cleaned_key:
             raise Exception("Failed to upload cleaned dataset to S3")

        await update_dataset(
            dataset_id,
            status="completed",
            cleanedDataPath=None, # Cleaned path is strictly S3 now
            cleanedUrl=cleaned_key,
            rows=df_final.shape[0],
            columns=df_final.shape[1]
        )
        
        log.info("processing_completed_success")

    except Exception as e:
        log.error("processing_failed_error", exc_info=True)
        await update_dataset(dataset_id, status="failed")

        # Attempt cleanup of raw dataset even on failure
        try:
             dataset = await get_dataset(dataset_id)
             if dataset and dataset.rawDataPath and not os.path.exists(dataset.rawDataPath):
                 delete_file(dataset.rawDataPath)
        except Exception:
             pass

        raise
    
    finally:
        # Robust local cleanup using glob patterns
        # We clean EVERYTHING related to this dataset_id in storage directories
        # This catches graphs, summary stats, raw/cleaned csvs, etc.
        try:
             patterns = [
                 # Dataset files (raw, cleaned)
                 f"storage/datasets/{dataset_id}_*",
                 # Graphs & Stats (png, json)
                 f"storage/graphs/{dataset_id}_*",
             ]
             
             import glob
             files_to_remove = []
             for pattern in patterns:
                 files_to_remove.extend(glob.glob(pattern))
             
             # Also include specific files tracked (though likely covered by glob)
             files_to_remove.extend(files_to_clean)
             
             # Deduplicate
             files_to_remove = list(set(files_to_remove))

             for path in files_to_remove:
                 if os.path.exists(path):
                     try:
                         os.remove(path)
                     except OSError:
                         log.warning("failed_to_clean_local_file", path=path)
        except Exception as e:
             log.error("cleanup_routine_failed", error=str(e))
        
        try:
             if 'dataset' in locals() and dataset and dataset.rawDataPath:
                 if not os.path.exists(dataset.rawDataPath): # Valid S3 key check (naive but consistent with rest)
                     delete_file(dataset.rawDataPath)
        except Exception:
             pass


@app.post("/dataset/upload", response_model=DatasetResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    dataset_url: Optional[str] = None,
    collaboration_id: Optional[str] = None,
    mode: str = "fast",
    user: JWTPayload = Depends(rate_limit_dependency),
):
    # Get user_id from authenticated user
    user_id = user.user_id
    if not file and not dataset_url:
        raise HTTPException(status_code=400, detail="Either file or dataset_url must be provided")
    
    if mode not in ["fast", "smart", "deep"]:
        raise HTTPException(status_code=400, detail="Mode must be one of: fast, smart, deep")
    
    if mode == "smart":
        usage_count = await count_user_mode_usage(user_id, "smart")
        if usage_count >= 3:
            raise HTTPException(
                status_code=429,
                detail="Smart mode limit reached (3 per user). Please use fast mode or wait."
            )
    elif mode == "deep":
        usage_count = await count_user_mode_usage(user_id, "deep")
        if usage_count >= 1:
            raise HTTPException(
                status_code=429,
                detail="Deep mode limit reached (1 per user). Please use fast or smart mode."
            )
    
    file_path = None
    database_name = None
    try:
        if file:
            filename = file.filename or "dataset"
            database_name = Path(filename).stem
            file_path = STORAGE_DIR / f"temp_{uuid.uuid4()}_{file.filename}"
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            df = await data_loader.load_from_file(str(file_path))
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                file_path = None
        else:
            if dataset_url:
                url_path = Path(dataset_url.split('?')[0])
                database_name = url_path.stem if url_path.suffix else None
                if not database_name or database_name in ['', 'index', 'default']:
                    database_name = "dataset_from_url"
            df = await data_loader.load_from_url(dataset_url)
        
        if df.shape[0] > 100000 or df.shape[1] > 100:
            raise HTTPException(
                status_code=400, 
                detail="Dataset too large. Maximum 100,000 rows and 100 columns allowed."
            )
        
        dataset = await create_dataset(
            rows=df.shape[0],
            columns=df.shape[1],
            status="uploaded",
            user_id=user_id,
            collaboration_id=collaboration_id,
            raw_data_path=None,
            database_name=database_name,
            mode=mode,
        )
        
        dataset_id = dataset.id
        
        await create_user_mode_usage(user_id=user_id, mode=mode, dataset_id=dataset_id)
        
        # Save raw dataset to S3
        temp_raw_path = data_loader.save_raw_dataset(df, dataset_id)
        s3_raw_key = f"datasets/{dataset_id}/raw.csv"
        raw_key = upload_file_and_get_key(temp_raw_path, s3_raw_key)
        
        # Cleanup local temp file unconditionally
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)

        # If upload failed, fail the request
        if not raw_key:
             await update_dataset(dataset_id, status="failed")
             raise HTTPException(status_code=500, detail="Failed to upload raw dataset to S3")

        await update_dataset(
            dataset_id,
            rawDataPath=raw_key,
            rawUrl=None,
        )

        # Choose queue name based on mode so heavy "deep" jobs
        # don't starve lighter ones.
        if mode == "deep":
            queue_name = "eda_deep"
            timeout = 60 * 30  # 30 minutes
        else:
            queue_name = "eda_fast"
            timeout = 60 * 10  # 10 minutes

        # Enqueue processing via Redis/RQ if Redis is configured, otherwise
        # fall back to FastAPI in-process background task.
        try:
            queue = get_queue(queue_name)
            # Use dataset_id as job_id so jobs are idempotent per dataset
            queue.enqueue(
                "jobs.eda_worker.process_dataset_job",
                dataset_id,
                mode,
                job_id=str(dataset_id),
                description=f"Process dataset {dataset_id} in {mode} mode",
                timeout=timeout,
                result_ttl=0,           # don't persist successful results
                failure_ttl=60 * 60*24, # keep failures for 1 day
                retry=Retry(
                    max=3,
                    interval=[60, 120, 300],  # backoff: 1m, 2m, 5m
                ),
            )
            logger.info("job_enqueued", dataset_id=dataset_id, mode=mode, queue=queue_name)
        except Exception:
            # If Redis is not available or enqueuing fails, we still don't want
            # to break the upload flow – use the in-process background task.
            background_tasks.add_task(process_dataset, dataset_id, mode)
        return DatasetResponse(
            dataset_id=dataset_id,
            rows=df.shape[0],
            columns=df.shape[1],
            status="uploaded",
            dataset_name=dataset.databaseName,
            mode=mode
        )
        
    except Exception as e:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")


@app.get("/dataset/{dataset_id}/raw")
async def get_raw_dataset(
    dataset_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization: check if user owns this dataset or is part of the collaboration
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    if dataset.rawDataPath and os.path.exists(dataset.rawDataPath):
        return FileResponse(
            dataset.rawDataPath,
            media_type="text/csv",
            filename=f"{dataset_id}_raw.csv"
        )

    if dataset.rawUrl:
        url = generate_presigned_url(dataset.rawUrl)
        if url:
            return JSONResponse({"url": url})

    raise HTTPException(status_code=404, detail="Raw dataset not available")


@app.get("/dataset/{dataset_id}/status", response_model=DatasetStatusResponse)
async def poll_dataset_status(
    dataset_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    """
    Poll the processing status of a specific dataset.
    
    This endpoint is designed for client-side polling to track dataset processing progress.
    Returns current status and metadata for the dataset.
    
    Status values:
    - "uploaded": Dataset received and queued for processing
    - "processing": Background processing in progress
    - "completed": All processing steps finished successfully
    - "failed": Processing encountered an error
    """
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization check
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    progress_info = None
    if dataset.status == "uploaded":
        progress_info = "Dataset uploaded successfully, queued for processing"
    elif dataset.status == "processing":
        progress_info = "Processing dataset: cleaning, analysis, and feature engineering in progress"
    elif dataset.status == "completed":
        progress_info = "Dataset processing completed successfully"
    elif dataset.status == "failed":
        progress_info = "Dataset processing failed - check logs for details"
    
    return DatasetStatusResponse(
        dataset_id=dataset.id,
        status=dataset.status,
        rows=dataset.rows,
        columns=dataset.columns,
        user_id=dataset.userId,
        collaboration_id=dataset.collaborationId,
        created_at=dataset.createdAt.isoformat(),
        updated_at=dataset.updatedAt.isoformat(),
        progress_info=progress_info,
        dataset_name=dataset.databaseName,
        mode=dataset.mode
    )


@app.get("/dataset/{dataset_id}/cleaned")
async def get_cleaned_dataset(
    dataset_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization check
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    if dataset.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Dataset processing not completed. Current status: {dataset.status}"
        )
    
    if dataset.cleanedDataPath and os.path.exists(dataset.cleanedDataPath):
        return FileResponse(
            dataset.cleanedDataPath,
            media_type="text/csv",
            filename=f"{dataset_id}_cleaned.csv"
        )

    if dataset.cleanedUrl:
        url = generate_presigned_url(dataset.cleanedUrl)
        if url:
            return JSONResponse({"url": url})

    raise HTTPException(status_code=404, detail="Cleaned dataset not available")


@app.get("/dataset/{dataset_id}/eda/graphs", response_model=GraphsResponse)
async def get_eda_graphs(
    dataset_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization check
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    graphs_metadata = await get_graphs_metadata(dataset_id)
    
    graphs = []
    for graph in graphs_metadata:
        graph_url = f"/dataset/{dataset_id}/graph/{graph.id}"
        graphs.append(GraphInfo(
            type=graph.type,
            column=graph.column,
            url=graph_url
        ))
    
    return GraphsResponse(graphs=graphs)


@app.get("/dataset/{dataset_id}/graph/{graph_id}")
async def get_graph_file(
    dataset_id: str,
    graph_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    # First verify the dataset exists and user has access
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization check
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    graph = await get_graph_metadata_by_id(graph_id)

    if not graph or graph.datasetId != dataset_id:
        raise HTTPException(status_code=404, detail="Graph not found")

    if not graph.filePath:
        raise HTTPException(status_code=404, detail="Graph file path/key not set")

    if os.path.exists(graph.filePath):
        return FileResponse(graph.filePath, media_type="image/png")

    url = generate_presigned_url(graph.filePath)
    if not url:
        raise HTTPException(status_code=500, detail="Failed to generate graph URL")

    return JSONResponse({"url": url})


@app.get("/dataset/{dataset_id}/features", response_model=FeaturesResponse)
async def get_features(
    dataset_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization check
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    if dataset.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Dataset processing not completed. Current status: {dataset.status}"
        )
    
    df = None
    
    if dataset.cleanedUrl:
        url = generate_presigned_url(dataset.cleanedUrl)
        if url:
            try:
                df = await data_loader.load_from_url(url)
            except Exception:
                df = None
    
    if df is None and dataset.cleanedDataPath and os.path.exists(dataset.cleanedDataPath):
        df = data_loader.load_saved_dataset(dataset.cleanedDataPath)
    
    if df is None:
        raise HTTPException(status_code=404, detail="Cleaned dataset not found")
    
    feature_service = FeatureEngineeringService()
    features = feature_service.get_input_output_features(df)
    
    return FeaturesResponse(
        input_features=features["input_features"],
        output_features=features["output_features"]
    )


@app.get("/dataset/{dataset_id}/logs/cleaning")
async def get_cleaning_logs_endpoint(
    dataset_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization check
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    logs = await get_cleaning_logs(dataset_id)
    return [{
        "column": log.column,
        "null_count": log.nullCount,
        "action": log.action,
        "reason": log.reason,
        "created_at": log.createdAt.isoformat()
    } for log in logs]


@app.get("/dataset/{dataset_id}/logs/outliers")
async def get_outlier_logs_endpoint(
    dataset_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization check
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    logs = await get_outlier_logs(dataset_id)
    return [{
        "column": log.column,
        "outlier_count": log.outlierCount,
        "method": log.method,
        "action": log.action,
        "created_at": log.createdAt.isoformat()
    } for log in logs]


@app.get("/dataset/{dataset_id}/logs/features")
async def get_feature_logs_endpoint(
    dataset_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization check
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    logs = await get_feature_logs(dataset_id)
    return [{
        "action": log.action,
        "details": log.details,
        "created_at": log.createdAt.isoformat()
    } for log in logs]


@app.get("/dataset/{dataset_id}/report")
async def get_cleaning_report_endpoint(
    dataset_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Authorization check
    if dataset.userId != user.user_id and dataset.collaborationId is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    if dataset.mode != "deep":
        raise HTTPException(
            status_code=400,
            detail=f"Cleaning report is only available for deep mode. This dataset was processed with {dataset.mode} mode."
        )
    
    report = await get_cleaning_report(dataset_id)
    if not report:
        raise HTTPException(status_code=404, detail="Cleaning report not found")
    
    return CleaningReportResponse(
        dataset_id=dataset_id,
        mode=dataset.mode,
        reasoning=report.reasoning,
        summary=report.summary,
        recommendations=report.recommendations,
        created_at=report.createdAt.isoformat()
    )


@app.get("/collaboration/{collaboration_id}/graphs", response_model=CollaborationGraphsResponse)
async def get_collaboration_graphs(
    collaboration_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    # Note: For collaboration endpoints, we allow access if user is authenticated
    # and part of the collaboration. The backend should verify collaboration membership.
    datasets = await list_datasets(collaboration_id=collaboration_id)
    if not datasets:
        return CollaborationGraphsResponse(graphs=[])

    graphs: List[CollaborationGraphInfo] = []
    for ds in datasets:
        ds_graphs = await get_graphs_metadata(ds.id)
        for g in ds_graphs:
            url = f"/dataset/{ds.id}/graph/{g.id}"
            graphs.append(
                CollaborationGraphInfo(
                    dataset_id=ds.id,
                    type=g.type,
                    column=g.column,
                    url=url,
                )
            )

    return CollaborationGraphsResponse(graphs=graphs)


@app.get("/collaboration/{collaboration_id}/datasets/cleaned", response_model=List[CollaborationDatasetItem])
async def get_collaboration_cleaned_datasets(
    collaboration_id: str,
    request: Request,
    user: JWTPayload = Depends(get_current_user),
):
    # Note: For collaboration endpoints, we allow access if user is authenticated
    # The backend should verify collaboration membership.
    datasets = await list_datasets(collaboration_id=collaboration_id)
    cleaned: List[CollaborationDatasetItem] = []
    
    base_url = str(request.base_url).rstrip('/')
    
    for ds in datasets:
        if ds.status != "completed":
            continue
        
        dataset_url = f"{base_url}/dataset/{ds.id}/cleaned"
        
        cleaned.append(
            CollaborationDatasetItem(
                id=ds.id,
                rows=ds.rows,
                columns=ds.columns,
                status=ds.status,
                user_id=ds.userId,
                collaboration_id=ds.collaborationId,
                url=dataset_url,
                dataset_name=ds.databaseName,
            )
        )
    return cleaned


@app.get("/datasets/status", response_model=List[DatasetStatusResponse])
async def poll_multiple_datasets_status(
    collaboration_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    user: JWTPayload = Depends(get_current_user),
):
    """
    Poll the processing status of multiple datasets.
    
    Useful for dashboard interfaces that need to show status of multiple datasets.
    Can filter by collaboration_id and/or status.
    
    Args:
        collaboration_id: Filter datasets by collaboration
        status_filter: Filter by specific status (uploaded, processing, completed, failed)
    """
    # Use authenticated user's ID - users can only see their own datasets
    user_id = user.user_id
    datasets = await list_datasets(user_id=user_id, collaboration_id=collaboration_id)
    
    if status_filter:
        datasets = [ds for ds in datasets if ds.status == status_filter]
    
    status_responses = []
    for dataset in datasets:
        progress_info = None
        if dataset.status == "uploaded":
            progress_info = "Dataset uploaded successfully, queued for processing"
        elif dataset.status == "processing":
            progress_info = "Processing dataset: cleaning, analysis, and feature engineering in progress"
        elif dataset.status == "completed":
            progress_info = "Dataset processing completed successfully"
        elif dataset.status == "failed":
            progress_info = "Dataset processing failed - check logs for details"
        
        status_responses.append(DatasetStatusResponse(
            dataset_id=dataset.id,
            status=dataset.status,
            rows=dataset.rows,
            columns=dataset.columns,
            user_id=dataset.userId,
            collaboration_id=dataset.collaborationId,
            created_at=dataset.createdAt.isoformat(),
            updated_at=dataset.updatedAt.isoformat(),
            progress_info=progress_info,
            dataset_name=dataset.databaseName,
            mode=dataset.mode
        ))
    
    return status_responses


@app.get("/datasets", response_model=List[DatasetItem])
async def list_datasets_endpoint(
    collaboration_id: Optional[str] = None,
    user: JWTPayload = Depends(get_current_user),
):
    # Use authenticated user's ID - users can only see their own datasets
    user_id = user.user_id
    datasets = await list_datasets(user_id=user_id, collaboration_id=collaboration_id)
    items: List[DatasetItem] = []
    for ds in datasets:
        items.append(
            DatasetItem(
                id=ds.id,
                rows=ds.rows,
                columns=ds.columns,
                status=ds.status,
                user_id=ds.userId,
                collaboration_id=ds.collaborationId,
                created_at=ds.createdAt.isoformat(),
                dataset_name=ds.databaseName,
                mode=ds.mode
            )
        )
    return items


@app.get("/")
async def root():
    return {
        "message": "NeuraLearn microservices api",
        "version": "1.0.0",
        "status": "running",
        "health": "healthy",
        "swagger": "/docs"
    }
