"""Tai va tien xu ly Breast Cancer Wisconsin (Diagnostic) tu UCI."""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/17/data.csv"
TARGET_COLUMN = "Diagnosis"
ID_COLUMN = "ID"


def download_dataset(destination: Path) -> Path:
    """Tai CSV chinh thuc tu UCI neu file chua ton tai."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(UCI_DATA_URL, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"UCI tra ve HTTP {response.status} khi tai dataset."
                )
            content = response.read()
    except (TimeoutError, URLError) as exc:
        raise RuntimeError(
            "Khong the tai dataset tu UCI. Hay kiem tra ket noi mang "
            f"hoac dat file CSV tai {destination}."
        ) from exc

    if not content:
        raise RuntimeError("UCI tra ve file dataset rong.")
    destination.write_bytes(content)
    return destination


def load_dataset(path: Path) -> pd.DataFrame:
    """Doc va kiem tra cac cot bat buoc cua dataset UCI."""
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay dataset: {path}")

    frame = pd.read_csv(path)
    required = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset thieu cot bat buoc: {names}")
    if frame.empty:
        raise ValueError("Dataset khong co dong du lieu.")

    labels = set(frame[TARGET_COLUMN].dropna().astype(str).str.strip().str.upper())
    if not labels.issubset({"B", "M"}) or not labels:
        raise ValueError(
            "Cot Diagnosis phai chi chua hai nhan B (benign) va M (malignant)."
        )
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Cot Diagnosis co nhan bi thieu.")
    return frame


def prepare_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Tach ID, features va nhan; quy uoc u=0, a=1 de tinh recall ung thu."""
    feature_columns = [
        column
        for column in frame.columns
        if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    if not feature_columns:
        raise ValueError("Dataset khong co cot dac trung.")

    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.dropna(axis=1, how="all")
    if features.empty:
        raise ValueError("Khong co dac trung so hop le trong dataset.")

    labels = (
        frame[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"B": 0, "M": 1})
        .astype("int64")
    )
    sample_ids = frame[ID_COLUMN].copy()
    return features, labels, sample_ids


def save_processed_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_ids: pd.Series,
    test_ids: pd.Series,
    destination: Path,
) -> tuple[Path, Path]:
    """Luu cac phan chia da ma hoa nhan; ID chi de doi chieu, khong la feature."""
    destination.mkdir(parents=True, exist_ok=True)

    def build_split(
        features: pd.DataFrame,
        labels: pd.Series,
        sample_ids: pd.Series,
    ) -> pd.DataFrame:
        split = features.copy()
        split.insert(0, ID_COLUMN, sample_ids.to_numpy())
        split[TARGET_COLUMN] = labels.to_numpy()
        return split

    train_path = destination / "wdbc_train.csv"
    test_path = destination / "wdbc_test.csv"
    build_split(X_train, y_train, train_ids).to_csv(train_path, index=False)
    build_split(X_test, y_test, test_ids).to_csv(test_path, index=False)
    return train_path, test_path
