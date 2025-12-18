import os
import json
from typing import Dict, Any, Optional
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
