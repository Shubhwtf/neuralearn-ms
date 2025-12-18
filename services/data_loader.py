import pandas as pd
import httpx
import os
from typing import Union
from pathlib import Path
import uuid


class DataLoader:
    def __init__(self, storage_dir: str = "storage/datasets"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def load_from_file(self, file_path: str) -> pd.DataFrame:
        file_path_obj = Path(file_path)
        
        if file_path_obj.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path_obj.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")
        
        return df
    
    async def load_from_url(self, url: str) -> pd.DataFrame:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
            if url.endswith('.csv'):
                import io
                df = pd.read_csv(io.StringIO(response.text))
            elif url.endswith('.json'):
                df = pd.read_json(response.text)
            else:
                try:
                    import io
                    df = pd.read_csv(io.StringIO(response.text))
                except:
                    df = pd.read_json(response.text)
            
            return df
    
    def save_raw_dataset(self, df: pd.DataFrame, dataset_id: str) -> str:
        file_path = self.storage_dir / f"{dataset_id}_raw.csv"
        df.to_csv(file_path, index=False)
        return str(file_path)
    
    def save_cleaned_dataset(self, df: pd.DataFrame, dataset_id: str) -> str:
        file_path = self.storage_dir / f"{dataset_id}_cleaned.csv"
        df.to_csv(file_path, index=False)
        return str(file_path)
    
    def load_saved_dataset(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
