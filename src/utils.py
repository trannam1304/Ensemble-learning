"""Cac ham hien thi thong tin cho lan chay thuc nghiem."""

from __future__ import annotations

import pandas as pd


def print_dataset_overview(
    features: pd.DataFrame,
    labels: pd.Series,
    missing_count: int,
) -> None:
    benign_count = int((labels == 0).sum())
    malignant_count = int((labels == 1).sum())
    print("=== Dataset: Breast Cancer Wisconsin (Diagnostic) ===")
    print(f"So mau: {len(features)} | So dac trung: {features.shape[1]}")
    print(f"Nhan B (benign): {benign_count} | Nhan M (malignant): {malignant_count}")
    print(f"So o dac trung bi thieu (duoc impute trong pipeline): {missing_count}")


def print_metrics_table(results: pd.DataFrame) -> None:
    columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "fit_seconds",
    ]
    display = results[columns].copy()
    print("\n=== Ket qua test (lop duong tinh = malignant) ===")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
