"""Chay pipeline thuc nghiem Ensemble Learning."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.evaluate import run_experiments, save_result_artifacts
from src.model import build_models
from src.preprocess import (
    download_dataset,
    load_dataset,
    prepare_dataset,
    save_processed_splits,
)
from src.utils import print_dataset_overview, print_metrics_table

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="So sanh base model va ensemble tren dataset UCI WDBC."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "wdbc.csv",
        help="CSV dataset (mac dinh: data/raw/wdbc.csv; tu dong tai neu chua co).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Thu muc ghi metrics va hinh (mac dinh: results).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Ti le tap test trong khoang (0, 1), mac dinh 0.2.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed de tai lap ket qua.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size phai nam trong khoang (0, 1).")

    if not args.data_path.exists():
        print(f"Khong tim thay CSV; dang tai dataset UCI vao {args.data_path}")
        download_dataset(args.data_path)
    raw_data = load_dataset(args.data_path)
    features, labels, sample_ids = prepare_dataset(raw_data)
    print_dataset_overview(features, labels, int(features.isna().sum().sum()))

    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        features,
        labels,
        sample_ids,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    train_path, test_path = save_processed_splits(
        X_train,
        X_test,
        y_train,
        y_test,
        train_ids,
        test_ids,
        PROJECT_ROOT / "data" / "processed",
    )
    print(f"Da luu train split: {train_path}")
    print(f"Da luu test split: {test_path}")

    models = build_models(random_state=args.random_state)
    results, predictions, scores = run_experiments(
        models, X_train, X_test, y_train, y_test
    )
    save_result_artifacts(
        results,
        predictions,
        scores,
        models,
        y_test,
        test_ids,
        list(features.columns),
        args.output_dir,
        dataset_rows=len(raw_data),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print_metrics_table(results)
    print(f"\nDa ghi ket qua va bieu do vao: {args.output_dir.resolve()}")
    print("Luu y: day la ket qua hoc tap, khong dung de chan doan y khoa.")


if __name__ == "__main__":
    main()
