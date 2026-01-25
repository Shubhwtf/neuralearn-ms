import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.gemini_client import GeminiClient


class DataCleaner:
    def __init__(self, mode: str = "fast"):
        self.mode = mode
        self.use_gemini = mode in ["smart", "deep"]
        self.max_gemini_calls = 3 if mode == "smart" else (3 if mode == "deep" else 0)
        self.gemini = GeminiClient() if self.use_gemini else None
        self.gemini_call_count = 0
        self.cleaning_logs: List[Dict[str, Any]] = []
        self.df_sample: Optional[pd.DataFrame] = None
    
    def get_column_stats(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        stats = {
            "dtype": str(df[column].dtype),
            "null_count": int(df[column].isnull().sum()),
            "unique_count": int(df[column].nunique()),
        }
        
        if df[column].dtype in ['int64', 'float64']:
            stats["min"] = float(df[column].min()) if not df[column].isnull().all() else None
            stats["max"] = float(df[column].max()) if not df[column].isnull().all() else None
            stats["mean"] = float(df[column].mean()) if not df[column].isnull().all() else None
            stats["median"] = float(df[column].median()) if not df[column].isnull().all() else None
            stats["std"] = float(df[column].std()) if not df[column].isnull().all() else None
        
        if df[column].dtype == 'object':
            mode_values = df[column].mode()
            stats["mode"] = mode_values.iloc[0] if len(mode_values) > 0 else None
        
        return stats
    
    def _get_strategy_heuristic(self, df: pd.DataFrame, column: str, null_count: int, total_rows: int) -> Dict[str, str]:
        null_percentage = null_count / total_rows if total_rows > 0 else 0
        dtype = df[column].dtype
        
        if null_percentage > 0.8:
            return {"action": "drop_column", "reason": f"Too many nulls ({null_percentage:.1%})"}
        
        if dtype in ['int64', 'float64']:
            std = df[column].std()
            if std == 0 or pd.isna(std):
                return {"action": "drop_column", "reason": "Zero variance"}
            
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            col_range = df[column].max() - df[column].min()
            
            if col_range > 0 and iqr / col_range < 0.3:
                return {"action": "median", "reason": "Outliers present, median more robust"}
            else:
                return {"action": "mean", "reason": "Normal distribution, mean appropriate"}
        else:
            unique_ratio = df[column].nunique() / total_rows if total_rows > 0 else 0
            if unique_ratio > 0.5:
                return {"action": "unknown", "reason": "High cardinality categorical"}
            else:
                return {"action": "mode", "reason": "Low cardinality, mode appropriate"}
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        self.cleaning_logs = []
        self.gemini_call_count = 0
        total_rows = len(df)
        
        if self.mode == "deep":
            self.df_sample = df.head(10).copy()
        elif self.mode == "smart":
            self.df_sample = df.head(10).copy()
        
        columns_with_nulls = []
        for column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                null_percentage = null_count / total_rows
                columns_with_nulls.append((column, null_count, null_percentage))
        
        columns_with_nulls.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        for column, null_count, null_percentage in columns_with_nulls:
            if self.mode == "fast":
                suggestion = self._get_strategy_heuristic(df, column, null_count, total_rows)
                action = suggestion["action"]
                reason = suggestion["reason"]
            elif self.mode == "smart":
                use_gemini_for_this = (
                    self.gemini_call_count < self.max_gemini_calls
                    and null_percentage > 0.2
                )
                if use_gemini_for_this:
                    stats = self.get_column_stats(df, column)
                    try:
                        suggestion = self.gemini.get_missing_value_strategy(
                            column_name=column,
                            dtype=stats["dtype"],
                            stats=stats,
                            null_count=null_count
                        )
                        action = suggestion["action"]
                        reason = suggestion["reason"]
                        self.gemini_call_count += 1
                    except Exception as e:
                        print(f"Gemini error for {column}: {e}, using heuristic")
                        suggestion = self._get_strategy_heuristic(df, column, null_count, total_rows)
                        action = suggestion["action"]
                        reason = suggestion["reason"]
                else:
                    suggestion = self._get_strategy_heuristic(df, column, null_count, total_rows)
                    action = suggestion["action"]
                    reason = suggestion["reason"]
            elif self.mode == "deep":
                use_gemini_for_this = (
                    self.gemini_call_count < self.max_gemini_calls
                    and null_percentage > 0.1
                )
                if use_gemini_for_this:
                    stats = self.get_column_stats(df, column)
                    try:
                        suggestion = self.gemini.get_missing_value_strategy(
                            column_name=column,
                            dtype=stats["dtype"],
                            stats=stats,
                            null_count=null_count
                        )
                        action = suggestion["action"]
                        reason = suggestion["reason"]
                        self.gemini_call_count += 1
                    except Exception as e:
                        print(f"Gemini error for {column}: {e}, using heuristic")
                        suggestion = self._get_strategy_heuristic(df, column, null_count, total_rows)
                        action = suggestion["action"]
                        reason = suggestion["reason"]
                else:
                    suggestion = self._get_strategy_heuristic(df, column, null_count, total_rows)
                    action = suggestion["action"]
                    reason = suggestion["reason"]
            else:
                suggestion = self._get_strategy_heuristic(df, column, null_count, total_rows)
                action = suggestion["action"]
                reason = suggestion["reason"]
            
            if action == "drop_column":
                df_cleaned = df_cleaned.drop(columns=[column])
                log_entry = {
                    "column": column,
                    "null_count": null_count,
                    "action": "Dropped column",
                    "reason": reason
                }
            elif action == "mean":
                mean_value = df[column].mean()
                original_dtype = df[column].dtype
                if pd.api.types.is_integer_dtype(original_dtype):
                    mean_value = int(np.round(mean_value))
                df_cleaned[column] = df_cleaned[column].fillna(original_dtype.type(mean_value))
                log_entry = {
                    "column": column,
                    "null_count": null_count,
                    "action": f"Filled with mean ({mean_value:.2f})",
                    "reason": reason
                }
            elif action == "median":
                median_value = df[column].median()
                original_dtype = df[column].dtype
                if pd.api.types.is_integer_dtype(original_dtype):
                    median_value = int(np.round(median_value))
                df_cleaned[column] = df_cleaned[column].fillna(original_dtype.type(median_value))
                log_entry = {
                    "column": column,
                    "null_count": null_count,
                    "action": f"Filled with median ({median_value:.2f})",
                    "reason": reason
                }
            elif action == "mode":
                mode_value = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else None
                if mode_value is not None:
                    df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                    log_entry = {
                        "column": column,
                        "null_count": null_count,
                        "action": f"Filled with mode ({mode_value})",
                        "reason": reason
                    }
                else:
                    df_cleaned[column] = df_cleaned[column].fillna("Unknown")
                    log_entry = {
                        "column": column,
                        "null_count": null_count,
                        "action": "Filled with 'Unknown'",
                        "reason": "No mode available, using 'Unknown'"
                    }
            elif action == "unknown":
                df_cleaned[column] = df_cleaned[column].fillna("Unknown")
                log_entry = {
                    "column": column,
                    "null_count": null_count,
                    "action": "Filled with 'Unknown'",
                    "reason": reason
                }
            elif action == "forward_fill":
                df_cleaned[column] = df_cleaned[column].ffill()
                log_entry = {
                    "column": column,
                    "null_count": null_count,
                    "action": "Forward filled",
                    "reason": reason
                }
            else:
                if df[column].dtype in ['int64', 'float64']:
                    median_value = df[column].median()
                    original_dtype = df[column].dtype
                    if pd.api.types.is_integer_dtype(original_dtype):
                        median_value = int(np.round(median_value))
                    df_cleaned[column] = df_cleaned[column].fillna(original_dtype.type(median_value))
                    log_entry = {
                        "column": column,
                        "null_count": null_count,
                        "action": f"Filled with median ({median_value:.2f})",
                        "reason": "Default strategy for numerical column"
                    }
                else:
                    mode_value = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else "Unknown"
                    df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                    log_entry = {
                        "column": column,
                        "null_count": null_count,
                        "action": f"Filled with mode ({mode_value})",
                        "reason": "Default strategy for categorical column"
                    }
            
            self.cleaning_logs.append(log_entry)
        
        return df_cleaned
    
    def get_cleaning_logs(self) -> List[Dict[str, Any]]:
        return self.cleaning_logs
    
    def get_sample_dataframe(self) -> Optional[pd.DataFrame]:
        return self.df_sample
