"""Danh gia mo hinh va xuat bang so lieu, bieu do, phan tich loi."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.model import majority_vote

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


def _positive_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Lay xac suat/diem cho lop malignant=1 de tinh ROC-AUC."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        class_positions = np.flatnonzero(model.classes_ == POSITIVE_LABEL)
        if len(class_positions) != 1:
            raise ValueError("Khong tim thay lop malignant=1 trong model.classes_.")
        return probabilities[:, class_positions[0]]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    raise TypeError(f"{type(model).__name__} khong ho tro scoring cho ROC-AUC.")


def _metric_row(
    model_name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    positive_scores: np.ndarray,
    fit_seconds: float,
) -> dict[str, float | str]:
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, positive_scores)),
        "fit_seconds": float(fit_seconds),
    }


def run_experiments(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit mo hinh chi tren train; tinh metric va luu prediction tren test."""
    fitted_models: dict[str, Any] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    scores_by_model: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        predicted_labels = np.asarray(model.predict(X_test))
        positive_scores = _positive_scores(model, X_test)

        fitted_models[model_name] = model
        predictions_by_model[model_name] = predicted_labels
        scores_by_model[model_name] = positive_scores
        rows.append(
            _metric_row(
                model_name, y_test, predicted_labels, positive_scores, fit_seconds
            )
        )

    voting_names = ("Decision Tree", "Logistic Regression", "Random Forest")
    if all(name in predictions_by_model for name in voting_names):
        start = time.perf_counter()
        base_predictions = np.column_stack(
            [predictions_by_model[name] for name in voting_names]
        )
        hard_predictions = majority_vote(base_predictions)
        vote_seconds = time.perf_counter() - start
        hard_scores = hard_predictions.astype(float)
        name = "Hard Voting (from scratch)"
        predictions_by_model[name] = hard_predictions
        scores_by_model[name] = hard_scores
        rows.append(
            _metric_row(name, y_test, hard_predictions, hard_scores, vote_seconds)
        )

    results = pd.DataFrame(rows).sort_values(
        ["f1", "roc_auc"], ascending=False, ignore_index=True
    )
    return results, predictions_by_model, scores_by_model


def _plot_metric_comparison(results: pd.DataFrame, output: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    figure, chart = plt.subplots(figsize=(15, 7))
    results.set_index("model")[metrics].plot(
        ax=chart,
        kind="bar",
        ylim=(0, 1.05),
        width=0.82,
        color=["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5", "#8064A2"],
    )
    chart.set_title("So sánh các chỉ số trên tập test", pad=18)
    chart.set_xlabel("")
    chart.set_ylabel("Điểm số")
    chart.legend(
        title="Chỉ số",
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.19),
    )
    chart.grid(axis="y", alpha=0.25)
    chart.tick_params(axis="x", labelrotation=35)
    for label in chart.get_xticklabels():
        label.set_ha("right")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_roc_curves(
    y_test: pd.Series, scores: dict[str, np.ndarray], output: Path
) -> None:
    plt.figure(figsize=(9, 7))
    for model_name, positive_scores in scores.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, positive_scores)
        area = auc(false_positive_rate, true_positive_rate)
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=1.8,
            label=f"{model_name} (AUC={area:.3f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ngẫu nhiên")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves - malignant là lớp dương")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrices(
    y_test: pd.Series, predictions: dict[str, np.ndarray], output: Path
) -> None:
    model_names = list(predictions)
    columns = 4
    rows = int(np.ceil(len(model_names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.8 * rows))
    flattened_axes = np.asarray(axes).reshape(-1)

    for axis, model_name in zip(flattened_axes, model_names):
        matrix = confusion_matrix(
            y_test,
            predictions[model_name],
            labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        )
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(model_name, fontsize=9)
        axis.set_xticks([0, 1], labels=["B", "M"])
        axis.set_yticks([0, 1], labels=["B", "M"])
        axis.set_xlabel("Du doan")
        axis.set_ylabel("Thuc te")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in flattened_axes[len(model_names) :]:
        axis.set_visible(False)
    figure.suptitle("Confusion matrix tren tap test (B=benign, M=malignant)")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _feature_importances(model: Any, feature_names: list[str]) -> pd.Series | None:
    estimator = model.named_steps.get("classifier")
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return None
    return pd.Series(estimator.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )


def _plot_feature_importances(
    fitted_models: dict[str, Any], feature_names: list[str], output: Path
) -> pd.DataFrame:
    selected: list[tuple[str, pd.Series]] = []
    for model_name in ("Random Forest", "AdaBoost"):
        if model_name in fitted_models:
            importances = _feature_importances(
                fitted_models[model_name], feature_names
            )
            if importances is not None:
                selected.append((model_name, importances.head(12)))

    if not selected:
        raise ValueError("Khong trich xuat duoc feature importance tu RF/AdaBoost.")

    figure, axes = plt.subplots(1, len(selected), figsize=(8 * len(selected), 7))
    axes_array = np.atleast_1d(axes)
    export_rows: list[pd.DataFrame] = []
    for axis, (model_name, importances) in zip(axes_array, selected):
        display = importances.sort_values()
        axis.barh(display.index, display.values, color="#4472C4")
        axis.set_title(f"Feature importance - {model_name}")
        axis.set_xlabel("Importance")
        export_rows.append(
            importances.rename("importance")
            .rename_axis("feature")
            .reset_index()
            .assign(model=model_name)
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pd.concat(export_rows, ignore_index=True)


def _write_analysis(
    results: pd.DataFrame,
    output: Path,
    error_count: int,
    false_negative_count: int,
    false_positive_count: int,
) -> None:
    best = results.iloc[0]
    baseline = results[
        results["model"].isin(["Decision Tree", "Logistic Regression", "k-NN", "SVM (RBF)"])
    ]
    best_baseline = baseline.sort_values(
        ["f1", "roc_auc"], ascending=False
    ).iloc[0]
    ensemble_gain = float(best["f1"] - best_baseline["f1"])

    lines = [
        "# Nhận xét thực nghiệm",
        "",
        "Các chỉ số dưới đây được tính trên một tập test phân tầng 20%;",
        "đây là kết quả của một lần chia dữ liệu, không phải ước lượng tổng quát hóa.",
        "",
        f"- Mô hình có F1 cao nhất trong lần chạy này: **{best['model']}** "
        f"(F1={best['f1']:.3f}, Recall={best['recall']:.3f}, "
        f"ROC-AUC={best['roc_auc']:.3f}).",
        f"- Base model tốt nhất theo F1: **{best_baseline['model']}** "
        f"(F1={best_baseline['f1']:.3f}); chênh lệch F1 giữa hai mô hình là "
        f"{ensemble_gain:+.3f}. Kết quả này không khẳng định ensemble luôn tốt hơn.",
        "- Với bài toán sàng lọc, Recall của lớp malignant cần được xem xét cùng "
        "Precision vì bỏ sót ca malignant có chi phí cao; ROC-AUC đánh giá khả "
        "năng xếp hạng ở nhiều ngưỡng.",
        f"- Random Forest dự đoán sai {error_count} mẫu trên test, gồm "
        f"{false_negative_count} false negative (M bị đoán thành B) và "
        f"{false_positive_count} false positive (B bị đoán thành M). Xem "
        "`results/misclassified_random_forest.csv` để kiểm tra từng trường hợp.",
        "- Feature importance của cây là mức đóng góp theo tiêu chí chia trong "
        "các cây, không chứng minh quan hệ nhân quả và có thể thiên lệch với "
        "các đặc trưng tương quan.",
        "- Ensemble thường giảm phương sai (Random Forest), kết hợp xác suất "
        "(Soft Voting/Stacking) hoặc tuần tự tập trung vào mẫu khó (AdaBoost), "
        "nhưng có thể tốn thời gian hơn và không đảm bảo thắng trên mọi lần chia.",
        "- Các mô hình chỉ phục vụ học tập trên bộ dữ liệu công khai; không dùng "
        "để chẩn đoán hay thay thế ý kiến chuyên gia y tế.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_result_artifacts(
    results: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    fitted_models: dict[str, Any],
    y_test: pd.Series,
    sample_ids: pd.Series,
    feature_names: list[str],
    output_dir: Path,
    *,
    dataset_rows: int,
    test_size: float,
    random_state: int,
) -> None:
    """Ghi metric, do thi, feature importance va danh sach loi Random Forest."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "metrics.csv", index=False, float_format="%.6f")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "name": "Breast Cancer Wisconsin (Diagnostic)",
                    "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
                    "instances": dataset_rows,
                    "features": len(feature_names),
                    "target": "Diagnosis (B=0 benign, M=1 malignant)",
                },
                "split": {
                    "test_size": test_size,
                    "random_state": random_state,
                    "stratified": True,
                },
                "positive_label": "M (malignant)",
                "models": results.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    _plot_metric_comparison(results, figure_dir / "model_comparison.png")
    _plot_roc_curves(y_test, scores, figure_dir / "roc_curves.png")
    _plot_confusion_matrices(
        y_test, predictions, figure_dir / "confusion_matrices.png"
    )
    importances = _plot_feature_importances(
        fitted_models, feature_names, figure_dir / "feature_importance.png"
    )
    importances.to_csv(
        output_dir / "feature_importance.csv", index=False, float_format="%.8f"
    )

    random_forest_predictions = predictions["Random Forest"]
    incorrect = random_forest_predictions != y_test.to_numpy()
    errors = pd.DataFrame(
        {
            "ID": sample_ids.to_numpy()[incorrect],
            "actual": np.where(y_test.to_numpy()[incorrect] == 1, "M", "B"),
            "predicted": np.where(random_forest_predictions[incorrect] == 1, "M", "B"),
        }
    )
    errors.to_csv(output_dir / "misclassified_random_forest.csv", index=False)
    false_negative_count = int(
        ((errors["actual"] == "M") & (errors["predicted"] == "B")).sum()
    )
    false_positive_count = int(
        ((errors["actual"] == "B") & (errors["predicted"] == "M")).sum()
    )
    _write_analysis(
        results,
        output_dir / "experiment_notes.md",
        len(errors),
        false_negative_count,
        false_positive_count,
    )
