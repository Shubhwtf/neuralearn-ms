from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
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
)

app = FastAPI(title="NeuraLearn microservices api", version="1.0.0")

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


class DatasetResponse(BaseModel):
    dataset_id: str
    rows: int
    columns: int
    status: str


class DatasetItem(BaseModel):
    id: str
    rows: int
    columns: int
    status: str
    user_id: str
    collaboration_id: Optional[str]
    created_at: str


class GraphInfo(BaseModel):
    type: str
    column: Optional[str]
    url: str


class GraphsResponse(BaseModel):
    graphs: List[GraphInfo]


class FeaturesResponse(BaseModel):
    input_features: List[str]
    output_features: List[str]


class CollaborationGraphInfo(BaseModel):
    dataset_id: str
    type: str
    column: Optional[str]
    url: str


class CollaborationGraphsResponse(BaseModel):
    graphs: List[CollaborationGraphInfo]


class CollaborationDatasetItem(BaseModel):
    id: str
    rows: int
    columns: int
    status: str
    user_id: str
    collaboration_id: Optional[str]
    url: str


async def process_dataset(dataset_id: str, df: pd.DataFrame):
    try:
        await update_dataset(dataset_id, status="processing")
        
        cleaner = DataCleaner()
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
            await create_graph_metadata(
                dataset_id=dataset_id,
                graph_type=graph["type"],
                column=graph.get("column"),
                file_path=graph["file_path"],
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
        
        cleaned_path = data_loader.save_cleaned_dataset(df_final, dataset_id)
        
        await update_dataset(
            dataset_id,
            status="completed",
            cleanedDataPath=cleaned_path,
            rows=df_final.shape[0],
            columns=df_final.shape[1]
        )
        
    except Exception as e:
        await update_dataset(dataset_id, status="failed")
        print(f"Error processing dataset {dataset_id}: {e}")


@app.post("/dataset/upload", response_model=DatasetResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    dataset_url: Optional[str] = None,
    user_id: str = "anonymous",
    collaboration_id: Optional[str] = None,
):
    if not file and not dataset_url:
        raise HTTPException(status_code=400, detail="Either file or dataset_url must be provided")
    
    try:
        if file:
            file_path = STORAGE_DIR / f"temp_{uuid.uuid4()}_{file.filename}"
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            df = await data_loader.load_from_file(str(file_path))
            os.remove(file_path)
        else:
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
        )
        
        dataset_id = dataset.id
        
        raw_path = data_loader.save_raw_dataset(df, dataset_id)
        
        await update_dataset(dataset_id, rawDataPath=raw_path)
        
        background_tasks.add_task(process_dataset, dataset_id, df)
        
        return DatasetResponse(
            dataset_id=dataset_id,
            rows=df.shape[0],
            columns=df.shape[1],
            status="uploaded"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")


@app.get("/dataset/{dataset_id}/raw")
async def get_raw_dataset(dataset_id: str):
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if not dataset.rawDataPath:
        raise HTTPException(status_code=404, detail="Raw dataset file not found")
    
    if not os.path.exists(dataset.rawDataPath):
        raise HTTPException(status_code=404, detail="Raw dataset file does not exist")
    
    return FileResponse(
        dataset.rawDataPath,
        media_type="text/csv",
        filename=f"{dataset_id}_raw.csv"
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
    
    if not dataset.cleanedDataPath:
        raise HTTPException(status_code=404, detail="Cleaned dataset file not found")
    
    if not os.path.exists(dataset.cleanedDataPath):
        raise HTTPException(status_code=404, detail="Cleaned dataset file does not exist")
    
    return FileResponse(
        dataset.cleanedDataPath,
        media_type="text/csv",
        filename=f"{dataset_id}_cleaned.csv"
    )


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
        raise HTTPException(status_code=404, detail="Graph file path/URL not set")

    if os.path.exists(graph.filePath):
        return FileResponse(graph.filePath, media_type="image/png")

    return JSONResponse({"url": graph.filePath})


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
    
    if not dataset.cleanedDataPath:
        raise HTTPException(status_code=404, detail="Cleaned dataset not found")
    
    df = data_loader.load_saved_dataset(dataset.cleanedDataPath)
    
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
            )
        )
    return cleaned


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
