from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import pandas as pd
import uuid
import os
from pathlib import Path

from services.data_loader import DataLoader
from services.cleaner import DataCleaner
from services.eda import EDAService
from services.outliers import OutlierService
from services.feature_engineering import FeatureEngineeringService
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
from s3_client import upload_file_and_get_key, generate_presigned_url
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

app = FastAPI(title="NeuraLearn microservices api", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


async def process_dataset(dataset_id: str, df: pd.DataFrame, mode: str = "fast"):
    try:
        await update_dataset(dataset_id, status="processing")
        
        dataset = await get_dataset(dataset_id)
        if not dataset:
            return
        
        cleaner = DataCleaner(mode=mode)
        df_cleaned = cleaner.handle_missing_values(df)
        
        for log in cleaner.get_cleaning_logs():
            await create_cleaning_log(
                dataset_id=dataset_id,
                column=log["column"],
                null_count=log["null_count"],
                action=log["action"],
                reason=log["reason"]
            )
        
        eda_results = eda_service.generate_all_eda(df_cleaned, dataset_id)

        for graph in eda_results["graphs"]:
            local_path = graph["file_path"]
            s3_key = f"graphs/{dataset_id}/{os.path.basename(local_path)}"
            stored_key = upload_file_and_get_key(local_path, s3_key) or local_path

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
        
        for log in outlier_service.get_outlier_logs():
            await create_outlier_log(
                dataset_id=dataset_id,
                column=log["column"],
                outlier_count=log["outlier_count"],
                method=log["method"],
                action=log["action"]
            )
        
        feature_service = FeatureEngineeringService()
        df_final = feature_service.apply_feature_engineering(df_cleaned)
        
        for log in feature_service.get_feature_logs():
            await create_feature_log(
                dataset_id=dataset_id,
                action=log["action"],
                details=log["details"]
            )
        
        if mode == "deep" and cleaner.gemini:
            try:
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
            except Exception as e:
                error_str = str(e)
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

        cleaned_s3_key = f"datasets/{dataset_id}/cleaned.csv"
        cleaned_key = upload_file_and_get_key(cleaned_path, cleaned_s3_key)
        
        await update_dataset(
            dataset_id,
            status="completed",
            cleanedDataPath=None if cleaned_key else cleaned_path,
            cleanedUrl=cleaned_key,
            rows=df_final.shape[0],
            columns=df_final.shape[1]
        )
        
    except Exception as e:
        await update_dataset(dataset_id, status="failed")
        raise


@app.post("/dataset/upload", response_model=DatasetResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    dataset_url: Optional[str] = None,
    user_id: str = "anonymous",
    collaboration_id: Optional[str] = None,
    mode: str = "fast",
):
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
        
        raw_path = data_loader.save_raw_dataset(df, dataset_id)
        await update_dataset(
            dataset_id,
            rawDataPath=raw_path,
            rawUrl=None,
        )
        
        background_tasks.add_task(process_dataset, dataset_id, df, mode)
        
        return DatasetResponse(
            dataset_id=dataset_id,
            rows=df.shape[0],
            columns=df.shape[1],
            status="uploaded",
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
async def get_raw_dataset(dataset_id: str):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
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
async def poll_dataset_status(dataset_id: str):
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
        database_name=dataset.databaseName,
        mode=dataset.mode
    )


@app.get("/dataset/{dataset_id}/cleaned")
async def get_cleaned_dataset(dataset_id: str):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
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
async def get_eda_graphs(dataset_id: str):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
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
async def get_graph_file(dataset_id: str, graph_id: str):
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
async def get_features(dataset_id: str):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
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
async def get_cleaning_logs_endpoint(dataset_id: str):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    logs = await get_cleaning_logs(dataset_id)
    return [{
        "column": log.column,
        "null_count": log.nullCount,
        "action": log.action,
        "reason": log.reason,
        "created_at": log.createdAt.isoformat()
    } for log in logs]


@app.get("/dataset/{dataset_id}/logs/outliers")
async def get_outlier_logs_endpoint(dataset_id: str):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    logs = await get_outlier_logs(dataset_id)
    return [{
        "column": log.column,
        "outlier_count": log.outlierCount,
        "method": log.method,
        "action": log.action,
        "created_at": log.createdAt.isoformat()
    } for log in logs]


@app.get("/dataset/{dataset_id}/logs/features")
async def get_feature_logs_endpoint(dataset_id: str):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    logs = await get_feature_logs(dataset_id)
    return [{
        "action": log.action,
        "details": log.details,
        "created_at": log.createdAt.isoformat()
    } for log in logs]


@app.get("/dataset/{dataset_id}/report")
async def get_cleaning_report_endpoint(dataset_id: str):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
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
async def get_collaboration_graphs(collaboration_id: str):
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
async def get_collaboration_cleaned_datasets(collaboration_id: str, request: Request):
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
                database_name=ds.databaseName,
            )
        )
    return cleaned


@app.get("/datasets/status", response_model=List[DatasetStatusResponse])
async def poll_multiple_datasets_status(
    user_id: Optional[str] = None,
    collaboration_id: Optional[str] = None,
    status_filter: Optional[str] = None
):
    """
    Poll the processing status of multiple datasets.
    
    Useful for dashboard interfaces that need to show status of multiple datasets.
    Can filter by user_id, collaboration_id, and/or status.
    
    Args:
        user_id: Filter datasets by user
        collaboration_id: Filter datasets by collaboration
        status_filter: Filter by specific status (uploaded, processing, completed, failed)
    """
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
            database_name=dataset.databaseName,
            mode=dataset.mode
        ))
    
    return status_responses


@app.get("/datasets", response_model=List[DatasetItem])
async def list_datasets_endpoint(
    user_id: Optional[str] = None,
    collaboration_id: Optional[str] = None,
):
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
                database_name=ds.databaseName,
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
