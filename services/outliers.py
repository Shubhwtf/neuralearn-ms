import pandas as pd
import numpy as np
from typing import Dict, List, Any


class OutlierService:
    def __init__(self):
        self.outlier_logs: List[Dict[str, Any]] = []
    
    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.Series:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        return outliers
    
    def detect_outliers_zscore(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
        return outliers
    
    def fix_outliers_cap(self, df: pd.DataFrame, column: str, outliers: pd.Series) -> pd.DataFrame:
        df_fixed = df.copy()
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        original_dtype = df[column].dtype
        
        if pd.api.types.is_integer_dtype(original_dtype):
            lower_bound = int(np.round(lower_bound))
            upper_bound = int(np.round(upper_bound))
        else:
            lower_bound = float(lower_bound)
            upper_bound = float(upper_bound)
        
        df_fixed.loc[df_fixed[column] < lower_bound, column] = original_dtype.type(lower_bound)
        df_fixed.loc[df_fixed[column] > upper_bound, column] = original_dtype.type(upper_bound)
        
        return df_fixed
    
    def fix_outliers_remove(self, df: pd.DataFrame, outliers: pd.Series) -> pd.DataFrame:
        return df[~outliers]
    
    def fix_outliers_replace_median(self, df: pd.DataFrame, column: str, outliers: pd.Series) -> pd.DataFrame:
        df_fixed = df.copy()
        median_value = df[column].median()
        
        original_dtype = df[column].dtype
        if pd.api.types.is_integer_dtype(original_dtype):
            median_value = int(np.round(median_value))
        else:
            median_value = float(median_value)
        
        df_fixed.loc[outliers, column] = original_dtype.type(median_value)
        return df_fixed
    
    def detect_and_fix_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = "IQR",
        fix_strategy: str = "cap",
        max_outlier_percentage: float = 0.1
    ) -> pd.DataFrame:
        df_fixed = df.copy()
        self.outlier_logs = []
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numerical_columns:
            if df[column].nunique() <= 1:
                continue
            
            if method == "IQR":
                outliers = self.detect_outliers_iqr(df_fixed, column)
            elif method == "Z-score":
                outliers = self.detect_outliers_zscore(df_fixed, column)
            else:
                continue
            
            outlier_count = outliers.sum()
            total_rows = len(df_fixed)
            outlier_percentage = outlier_count / total_rows if total_rows > 0 else 0
            
            if outlier_count == 0:
                continue
            
            if outlier_percentage > max_outlier_percentage:
                log_entry = {
                    "column": column,
                    "outlier_count": int(outlier_count),
                    "method": method,
                    "action": f"Skipped (outlier percentage {outlier_percentage:.1%} > {max_outlier_percentage:.1%})"
                }
                self.outlier_logs.append(log_entry)
                continue
            
            if fix_strategy == "cap":
                df_fixed = self.fix_outliers_cap(df_fixed, column, outliers)
                action = "Capped to upper/lower bound"
            elif fix_strategy == "remove":
                df_fixed = self.fix_outliers_remove(df_fixed, outliers)
                action = "Removed rows"
            elif fix_strategy == "replace_median":
                df_fixed = self.fix_outliers_replace_median(df_fixed, column, outliers)
                action = "Replaced with median"
            else:
                continue
            
            log_entry = {
                "column": column,
                "outlier_count": int(outlier_count),
                "method": method,
                "action": action
            }
            self.outlier_logs.append(log_entry)
        
        return df_fixed
    
    def get_outlier_logs(self) -> List[Dict[str, Any]]:
        return self.outlier_logs
