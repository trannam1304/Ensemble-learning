"""Khai bao base models, ensemble va hard voting minh hoa tu dau."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _pipeline(
    estimator: Any,
    *,
    scale: bool = False,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def build_models(random_state: int = 42) -> dict[str, Any]:
    """Tao cac baseline va ensemble; moi tien xu ly duoc fit tren train set."""
    tree = _pipeline(
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        )
    )
    logistic = _pipeline(
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        scale=True,
    )
    forest = _pipeline(
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
    )

    ada = _pipeline(
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
            n_estimators=100,
            learning_rate=0.5,
            random_state=random_state,
        )
    )
    knn = _pipeline(KNeighborsClassifier(n_neighbors=5), scale=True)
    calibrated_svm = CalibratedClassifierCV(
        estimator=Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("svc", SVC(C=1.0, kernel="rbf")),
            ]
        ),
        cv=5,
    )
    svm = Pipeline([("classifier", calibrated_svm)])
    gradient_boosting = _pipeline(
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=random_state,
        )
    )
    soft_voting = VotingClassifier(
        estimators=[
            ("tree", tree),
            ("logistic", logistic),
            ("forest", forest),
        ],
        voting="soft",
        weights=[1, 2, 2],
        n_jobs=1,
    )
    stacking = StackingClassifier(
        estimators=[
            ("tree", tree),
            ("logistic", logistic),
            ("forest", forest),
        ],
        final_estimator=LogisticRegression(max_iter=3000),
        cv=5,
        stack_method="predict_proba",
        n_jobs=1,
    )
    return {
        "Decision Tree": tree,
        "Logistic Regression": logistic,
        "k-NN": knn,
        "SVM (RBF)": svm,
        "Random Forest": forest,
        "AdaBoost": ada,
        "Gradient Boosting": gradient_boosting,
        "Soft Voting": soft_voting,
        "Stacking": stacking,
    }


def majority_vote(base_predictions: ArrayLike) -> NDArray[Any]:
    """Binh chon da so tu ma tran du doan (mau x base model).

    Neu so phieu hoa, chon nhan nho hon theo thu tu tang dan; voi B=0/M=1,
    quy tac nay uu tien nhan benign khi cac base model bat dong.
    """
    predictions = np.asarray(base_predictions)
    if predictions.ndim != 2:
        raise ValueError("Du doan dau vao phai la ma tran 2 chieu.")
    if predictions.shape[1] == 0:
        raise ValueError("Can it nhat mot base model de bo phieu.")
    if predictions.size == 0:
        return np.asarray([], dtype=predictions.dtype)

    classes = np.unique(predictions)
    vote_counts = np.stack(
        [(predictions == class_label).sum(axis=1) for class_label in classes],
        axis=1,
    )
    return classes[np.argmax(vote_counts, axis=1)]
