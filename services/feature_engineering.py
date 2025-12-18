import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from services.gemini_client import GeminiClient


class FeatureEngineeringService:
    def __init__(self, use_gemini: bool = True, min_columns_for_gemini: int = 5):
        self.use_gemini = use_gemini
        self.min_columns_for_gemini = min_columns_for_gemini
        self.gemini = GeminiClient() if use_gemini else None
        self.feature_logs: List[Dict[str, Any]] = []
        self.scalers = {}
        self.encoders = {}
    
    def get_sample_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    "dtype": "numerical",
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
            else:
                stats[col] = {
                    "dtype": "categorical",
                    "unique_count": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }
        return stats
    
    def encode_categorical(self, df: pd.DataFrame, column: str, method: str, reason: str = "") -> pd.DataFrame:
        df_encoded = df.copy()
        
        if method == "one-hot":
            dummies = pd.get_dummies(df_encoded[column], prefix=column)
            df_encoded = pd.concat([df_encoded.drop(columns=[column]), dummies], axis=1)
            self.feature_logs.append({
                "action": f"Encoded '{column}' (one-hot)",
                "details": reason or "Low cardinality categorical"
            })
        elif method == "label":
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            self.encoders[column] = le
            self.feature_logs.append({
                "action": f"Encoded '{column}' (label)",
                "details": reason or "High cardinality categorical"
            })
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, column: str, method: str, reason: str = "") -> pd.DataFrame:
        df_scaled = df.copy()
        
        if method == "standard":
            scaler = StandardScaler()
            df_scaled[column] = scaler.fit_transform(df_scaled[[column]])
            self.scalers[column] = scaler
            self.feature_logs.append({
                "action": f"Scaled '{column}' (standard)",
                "details": reason or "Different scales with other features"
            })
        elif method == "min-max":
            scaler = MinMaxScaler()
            df_scaled[column] = scaler.fit_transform(df_scaled[[column]])
            self.scalers[column] = scaler
            self.feature_logs.append({
                "action": f"Scaled '{column}' (min-max)",
                "details": reason or "Large value range"
            })
        
        return df_scaled
    
    def create_derived_features(self, df: pd.DataFrame, feature_specs: List[Dict[str, str]]) -> pd.DataFrame:
        df_new = df.copy()
        
        for spec in feature_specs:
            name = spec.get("name")
            description = spec.get("description", "")
            reason = spec.get("reason", description)
            
            if "age" in description.lower() or "age" in name.lower():
                if "Age" in df.columns:
                    df_new[name] = pd.cut(df["Age"], bins=[0, 18, 35, 50, 100], labels=["Young", "Adult", "Middle", "Senior"])
                    self.feature_logs.append({
                        "action": f"Created '{name}'",
                        "details": reason or description
                    })
            else:
                self.feature_logs.append({
                    "action": f"Suggested: create '{name}'",
                    "details": reason or description
                })
        
        return df_new
    
    def _get_heuristic_suggestions(self, df: pd.DataFrame) -> Dict[str, Any]:
        encoding = []
        scaling = []
        drop = []
        new_features = []
        
        for col in df.columns:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            total_rows = len(df)
            
            if dtype == 'object':
                if unique_count <= 10:
                    encoding.append({"column": col, "method": "one-hot", "reason": "Low cardinality"})
                else:
                    encoding.append({"column": col, "method": "label", "reason": "High cardinality"})
            
            if dtype in ['int64', 'float64']:
                std = df[col].std()
                if std > 0:
                    cv = std / abs(df[col].mean()) if df[col].mean() != 0 else 0
                    if cv > 1:
                        scaling.append({"column": col, "method": "standard", "reason": "High variance"})
            
            if unique_count <= 1:
                drop.append({"column": col, "reason": "Constant column"})
            elif unique_count == total_rows:
                drop.append({"column": col, "reason": "Unique identifier"})
        
        return {
            "encoding": encoding,
            "scaling": scaling,
            "drop": drop,
            "new_features": new_features
        }
    
    def apply_feature_engineering(
        self, 
        df: pd.DataFrame, 
        enable_scaling: bool = False,
        enable_encoding: bool = True,
        enable_dropping: bool = False
    ) -> pd.DataFrame:
        df_engineered = df.copy()
        self.feature_logs = []
        
        num_columns = len(df.columns)
        has_categorical = any(df[col].dtype == 'object' for col in df.columns)
        should_use_gemini = (
            self.use_gemini 
            and num_columns >= self.min_columns_for_gemini
            and (has_categorical or num_columns > 10)
        )
        
        if should_use_gemini:
            try:
                sample_stats = self.get_sample_statistics(df)
                suggestions = self.gemini.get_feature_engineering_suggestions(
                    columns=list(df.columns),
                    dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                    sample_data=sample_stats
                )
            except Exception as e:
                print(f"Gemini error: {e}, using heuristics")
                suggestions = self._get_heuristic_suggestions(df)
        else:
            suggestions = self._get_heuristic_suggestions(df)
        
        encoding_suggestions = suggestions.get("encoding", [])
        scaling_suggestions = suggestions.get("scaling", [])
        drop_suggestions = suggestions.get("drop", [])
        new_features = suggestions.get("new_features", [])
        
        for enc_spec in encoding_suggestions:
            col = enc_spec.get("column")
            method = enc_spec.get("method", "label")
            reason = enc_spec.get("reason", "Categorical encoding needed")
            if col in df_engineered.columns:
                if enable_encoding:
                    df_engineered = self.encode_categorical(df_engineered, col, method, reason)
                else:
                    self.feature_logs.append({
                        "action": f"Suggested: encode '{col}' ({method})",
                        "details": reason
                    })
        
        for scale_spec in scaling_suggestions:
            col = scale_spec.get("column")
            method = scale_spec.get("method", "standard")
            reason = scale_spec.get("reason", "Feature scaling needed")
            if col in df_engineered.columns and df_engineered[col].dtype in ['int64', 'float64']:
                if enable_scaling:
                    df_engineered = self.scale_features(df_engineered, col, method, reason)
                else:
                    self.feature_logs.append({
                        "action": f"Suggested: scale '{col}' ({method})",
                        "details": reason
                    })
        
        for drop_spec in drop_suggestions:
            if isinstance(drop_spec, dict):
                col = drop_spec.get("column")
                reason = drop_spec.get("reason", "Low importance")
            else:
                col = drop_spec
                reason = "Low importance"
            
            if col in df_engineered.columns:
                if enable_dropping:
                    df_engineered = df_engineered.drop(columns=[col])
                    self.feature_logs.append({
                        "action": f"Dropped '{col}'",
                        "details": reason
                    })
                else:
                    self.feature_logs.append({
                        "action": f"Suggested: drop '{col}'",
                        "details": reason
                    })
        
        if new_features:
            df_engineered = self.create_derived_features(df_engineered, new_features)
        
        return df_engineered
    
    def get_feature_logs(self) -> List[Dict[str, Any]]:
        return self.feature_logs
    
    def get_input_output_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        columns = list(df.columns)
        output_features = []
        input_features = []
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ["target", "label", "y", "output", "class"]):
                output_features.append(col)
            else:
                input_features.append(col)
        
        if not output_features and columns:
            output_features = [columns[-1]]
            input_features = columns[:-1]
        elif not output_features:
            input_features = columns
        
        return {
            "input_features": input_features,
            "output_features": output_features
        }
