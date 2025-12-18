import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class EDAService:
    def __init__(self, storage_dir: str = "storage/graphs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

    def generate_histogram(self, df: pd.DataFrame, column: str, dataset_id: str) -> str:
        plt.figure(figsize=(10, 6))
        df[column].hist(bins=30, edgecolor="black")
        plt.title(f"Histogram of {column}", fontsize=14, fontweight="bold")
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3)

        file_path = self.storage_dir / f"{dataset_id}_histogram_{column}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(file_path)

    def generate_boxplot(self, df: pd.DataFrame, column: str, dataset_id: str) -> str:
        plt.figure(figsize=(10, 6))
        df.boxplot(column=[column])
        plt.title(f"Boxplot of {column} (Outlier Detection)", fontsize=14, fontweight="bold")
        plt.ylabel(column, fontsize=12)
        plt.grid(True, alpha=0.3)

        file_path = self.storage_dir / f"{dataset_id}_boxplot_{column}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(file_path)

    def generate_correlation_heatmap(self, df: pd.DataFrame, dataset_id: str) -> str | None:
        numerical_df = df.select_dtypes(include=[np.number])
        if numerical_df.empty or len(numerical_df.columns) < 2:
            return None

        plt.figure(figsize=(12, 10))
        correlation_matrix = numerical_df.corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")

        file_path = self.storage_dir / f"{dataset_id}_correlation_heatmap.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(file_path)

    def generate_countplot(self, df: pd.DataFrame, column: str, dataset_id: str) -> str:
        plt.figure(figsize=(10, 6))
        value_counts = df[column].value_counts().head(20)
        value_counts.plot(kind="bar")
        plt.title(f"Count Plot of {column}", fontsize=14, fontweight="bold")
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        file_path = self.storage_dir / f"{dataset_id}_countplot_{column}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(file_path)

    def generate_all_eda(self, df: pd.DataFrame, dataset_id: str) -> Dict[str, Any]:
        graphs_metadata: List[Dict[str, Any]] = []

        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            hist_path = self.generate_histogram(df, col, dataset_id)
            box_path = self.generate_boxplot(df, col, dataset_id)
            graphs_metadata.append(
                {"type": "histogram", "column": col, "file_path": hist_path}
            )
            graphs_metadata.append(
                {"type": "boxplot", "column": col, "file_path": box_path}
            )

        if len(numerical_columns) >= 2:
            heatmap_path = self.generate_correlation_heatmap(df, dataset_id)
            if heatmap_path is not None:
                graphs_metadata.append(
                    {"type": "heatmap", "column": None, "file_path": heatmap_path}
                )

        categorical_columns = df.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            if df[col].nunique() <= 50:
                count_path = self.generate_countplot(df, col, dataset_id)
                graphs_metadata.append(
                    {"type": "countplot", "column": col, "file_path": count_path}
                )

        summary_stats = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numerical_summary": df.describe().to_dict()
            if not numerical_columns.empty
            else {},
            "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
            "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
        }

        stats_path = self.storage_dir / f"{dataset_id}_summary_stats.json"
        with open(stats_path, "w") as f:
            json.dump(summary_stats, f, indent=2, default=str)

        return {
            "graphs": graphs_metadata,
            "summary_stats": summary_stats,
            "stats_path": str(stats_path),
        }
