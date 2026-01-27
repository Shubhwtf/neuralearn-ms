from pydantic import BaseModel, Field
from typing import Optional, List


class DatasetResponse(BaseModel):
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    rows: int = Field(..., description="Number of rows in the dataset")
    columns: int = Field(..., description="Number of columns in the dataset")
    status: str = Field(..., description="Processing status: uploaded, processing, completed, or failed")
    dataset_name: Optional[str] = Field(None, description="Name of the dataset (extracted from filename or URL)")
    mode: Optional[str] = Field(None, description="Cleaning mode used: fast, smart, or deep")


class DatasetItem(BaseModel):
    id: str = Field(..., description="Unique identifier for the dataset")
    rows: int = Field(..., description="Number of rows in the dataset")
    columns: int = Field(..., description="Number of columns in the dataset")
    status: str = Field(..., description="Processing status")
    user_id: str = Field(..., description="User who uploaded the dataset")
    collaboration_id: Optional[str] = Field(None, description="Collaboration group ID if applicable")
    created_at: str = Field(..., description="ISO timestamp of when dataset was created")
    dataset_name: Optional[str] = Field(None, description="Name of the dataset (extracted from filename)")
    mode: Optional[str] = Field(None, description="Cleaning mode used: fast, smart, or deep")


class GraphInfo(BaseModel):
    type: str = Field(..., description="Type of graph: histogram, boxplot, heatmap, or countplot")
    column: Optional[str] = Field(None, description="Column name for the graph (if applicable)")
    url: str = Field(..., description="URL to access the graph")


class GraphsResponse(BaseModel):
    graphs: List[GraphInfo] = Field(..., description="List of available graphs for the dataset")


class FeaturesResponse(BaseModel):
    input_features: List[str] = Field(..., description="List of input feature column names")
    output_features: List[str] = Field(..., description="List of output/target feature column names")


class CollaborationGraphInfo(BaseModel):
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    type: str = Field(..., description="Type of graph")
    column: Optional[str] = Field(None, description="Column name for the graph")
    url: str = Field(..., description="URL to access the graph")


class CollaborationGraphsResponse(BaseModel):
    graphs: List[CollaborationGraphInfo] = Field(..., description="List of graphs across all datasets in collaboration")


class CollaborationDatasetItem(BaseModel):
    id: str = Field(..., description="Unique identifier for the dataset")
    rows: int = Field(..., description="Number of rows in the dataset")
    columns: int = Field(..., description="Number of columns in the dataset")
    status: str = Field(..., description="Processing status")
    user_id: str = Field(..., description="User who uploaded the dataset")
    collaboration_id: Optional[str] = Field(None, description="Collaboration group ID")
    url: str = Field(..., description="URL to download the cleaned dataset")
    dataset_name: Optional[str] = Field(None, description="Name of the dataset")


class DatasetStatusResponse(BaseModel):
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    status: str = Field(..., description="Current processing status")
    rows: int = Field(..., description="Number of rows in the dataset")
    columns: int = Field(..., description="Number of columns in the dataset")
    user_id: str = Field(..., description="User who uploaded the dataset")
    collaboration_id: Optional[str] = Field(None, description="Collaboration group ID")
    created_at: str = Field(..., description="ISO timestamp of creation")
    updated_at: str = Field(..., description="ISO timestamp of last update")
    progress_info: Optional[str] = Field(None, description="Human-readable progress information")
    dataset_name: Optional[str] = Field(None, description="Name of the dataset")
    mode: Optional[str] = Field(None, description="Cleaning mode used")


class CleaningReportResponse(BaseModel):
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    mode: str = Field(..., description="Cleaning mode used")
    reasoning: str = Field(..., description="Detailed explanation of cleaning decisions with examples")
    summary: str = Field(..., description="Executive summary of data quality assessment")
    recommendations: Optional[str] = Field(None, description="Actionable recommendations for data quality improvement")
    created_at: str = Field(..., description="ISO timestamp of report creation")

