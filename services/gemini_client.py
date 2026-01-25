import os
import json
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def get_missing_value_strategy(
        self, 
        column_name: str, 
        dtype: str, 
        stats: Dict[str, Any],
        null_count: int
    ) -> Dict[str, Any]:
        prompt = f"""
        You are a data cleaning expert. Analyze the following column and suggest the best strategy for handling missing values.

        Column Name: {column_name}
        Data Type: {dtype}
        Null Count: {null_count}
        Statistics: {json.dumps(stats, indent=2)}

        Please suggest ONE of the following strategies:
        1. "mean" - Fill with mean (for numerical continuous)
        2. "median" - Fill with median (for numerical with outliers)
        3. "mode" - Fill with mode (for categorical or discrete numerical)
        4. "unknown" - Fill with "Unknown" string (for categorical)
        5. "forward_fill" - Forward fill (for time series)
        6. "drop_column" - Drop the entire column (if too many nulls or useless)

        Respond ONLY with a JSON object in this exact format:
        {{
            "action": "median",
            "reason": "Brief explanation why this strategy is best"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            return {
                "action": result.get("action", "mean"),
                "reason": result.get("reason", "No reason provided")
            }
        except Exception as e:
            print(f"Gemini API error: {e}. Using default strategy.")
            if dtype in ['int64', 'float64']:
                return {"action": "median", "reason": "Default: median for numerical columns"}
            else:
                return {"action": "mode", "reason": "Default: mode for categorical columns"}
    
    def get_feature_engineering_suggestions(
        self, 
        columns: list, 
        dtypes: Dict[str, str],
        sample_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = f"""
        You are a feature engineering expert. Analyze this dataset and provide suggestions.

        Columns: {columns}
        Data Types: {json.dumps(dtypes, indent=2)}
        Sample Statistics: {json.dumps(sample_data, indent=2)}

        Provide suggestions for:
        1. Categorical encoding (which columns need encoding, what method: one-hot, label, etc.)
        2. Feature scaling (which columns need scaling: standard, min-max, etc.)
        3. Columns to drop (low importance or redundant)
        4. New derived features to create

        Respond ONLY with a JSON object in this exact format:
        {{
            "encoding": [
                {{"column": "Gender", "method": "one-hot", "reason": "Low cardinality categorical"}},
                {{"column": "Category", "method": "label", "reason": "High cardinality categorical"}}
            ],
            "scaling": [
                {{"column": "Age", "method": "standard", "reason": "Different scales with other features"}},
                {{"column": "Salary", "method": "min-max", "reason": "Large value range"}}
            ],
            "drop": [
                {{"column": "Column1", "reason": "Low variance"}},
                {{"column": "Column2", "reason": "Redundant with Column3"}}
            ],
            "new_features": [
                {{"name": "AgeGroup", "description": "Categorize age into groups", "reason": "Improves model interpretability"}}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            encoding = result.get("encoding", [])
            scaling = result.get("scaling", [])
            drop = result.get("drop", [])
            new_features = result.get("new_features", [])
            
            if isinstance(drop, list) and drop and isinstance(drop[0], str):
                drop = [{"column": col, "reason": "Low importance"} for col in drop]
            
            return {
                "encoding": encoding,
                "scaling": scaling,
                "drop": drop,
                "new_features": new_features
            }
        except Exception as e:
            print(f"Gemini API error: {e}. Using default suggestions.")
            return {
                "encoding": [],
                "scaling": [],
                "drop": [],
                "new_features": []
            }
    
    def get_deep_cleaning_report(
        self,
        df_sample: pd.DataFrame,
        cleaning_logs: List[Dict[str, Any]],
        dataset_name: str
    ) -> Dict[str, Any]:
        def convert_to_serializable(obj):
            if obj is None:
                return None
            try:
                if pd.isna(obj):
                    return None
            except (TypeError, ValueError):
                pass
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.int_)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32, np.float16, np.float_)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (str, int, float, bool)):
                return obj
            try:
                return str(obj)
            except Exception:
                return None
        
        serializable_logs = []
        for log in cleaning_logs[:10]:
            serializable_log = {k: convert_to_serializable(v) for k, v in log.items()}
            serializable_logs.append(serializable_log)
        cleaning_summary = []
        for log in serializable_logs:
            cleaning_summary.append({
                "column": log.get("column", "unknown")[:30],
                "action": log.get("action", "unknown")[:80],
                "null_count": log.get("null_count", 0),
                "reason": log.get("reason", "")[:100] if log.get("reason") else ""
            })
        
        prompt = f"""
        You are analyzing a dataset cleaning process. Provide a detailed report with examples.

        Dataset: {dataset_name}
        Cleaning Actions Performed: {json.dumps(cleaning_summary, indent=2, default=str)}

        Provide a detailed JSON report:
        {{
            "reasoning": "Detailed explanation (4-6 sentences) of why each cleaning action was chosen. Include specific examples with column names, numbers, and values. For example: 'Column Salary had 150 null values (15% of 1000 rows), so we filled them with median value 42,500 because the data distribution had outliers that would skew the mean. Column City had 80 nulls in categorical data, so we used mode 'Unknown' to preserve the data distribution without introducing bias.' Explain the statistical reasoning and impact of each decision with concrete numbers from the cleaning actions above.",
            "summary": "Comprehensive summary (3-4 sentences) of data quality issues found, what was fixed, and overall data quality improvement. Include specific numbers and examples from the cleaning actions. Mention the total number of columns cleaned and key improvements.",
            "recommendations": "3-4 actionable recommendations (1-2 sentences each) with specific examples. For instance: 'Review Column Age for domain-specific validation as 10% of values were filled with median 35. Consider checking if missing ages follow a pattern.' Be specific about which columns need attention and why."
        }}

        Be specific with examples and numbers from the cleaning actions above.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            recommendations = result.get("recommendations", "No recommendations provided")
            if isinstance(recommendations, list):
                recommendations = "\n".join(f"- {rec}" if isinstance(rec, str) else str(rec) for rec in recommendations)
            elif not isinstance(recommendations, str):
                recommendations = str(recommendations)
            
            return {
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "summary": result.get("summary", "No summary provided"),
                "recommendations": recommendations
            }
        except Exception as e:
            raise Exception(f"Failed to generate Gemini report: {str(e)}")