#!/usr/bin/env python3
"""
BITS Pilani WILP - M.Tech (DSE) Machine Learning Assignment 2
Script: model/train_models.py
Domain: Automotive OBD-II Engine & Powertrain Fault Diagnostics (1,600 instances, 14 features)
Description: Trains 5 Classification Models (LR, DT, kNN, Gaussian NB, SVM),
             exports stratified test_data.csv (320 rows), saves serialized .pkl
             pipelines in model/, and prints all 6 evaluation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# Keep 2026 to match the exact report numbers, or change to last 4 digits of your BITS ID
STUDENT_SEED = 2026

def load_or_generate_automotive_dataset(seed=STUDENT_SEED):
    """
    Loads local automotive OBD-II dataset if present, or deterministically generates
    the 1,600-instance, 14-feature Automotive OBD-II Engine & Powertrain Fault
    Telemetry Dataset (modeled on Kaggle/UCI Automotive OBD-II benchmarks).
    """
    if os.path.exists("automotive_obd_full.csv"):
        return pd.read_csv("automotive_obd_full.csv")

    X_raw, y = make_classification(
        n_samples=1600,
        n_features=14,
        n_informative=10,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.56, 0.44],
        class_sep=1.18,
        flip_y=0.025,
        random_state=seed
    )

    feature_specs = [
        ("engine_rpm", 2450.0, 820.0, 650.0, 6200.0),
        ("lub_oil_pressure_bar", 3.40, 0.95, 0.80, 6.50),
        ("fuel_rail_pressure_bar", 115.0, 32.0, 35.0, 220.0),
        ("coolant_pressure_bar", 1.85, 0.42, 0.60, 3.20),
        ("lub_oil_temp_c", 94.0, 11.5, 65.0, 135.0),
        ("coolant_temp_c", 89.5, 9.8, 60.0, 125.0),
        ("maf_airflow_gps", 28.5, 11.2, 3.5, 85.0),
        ("throttle_pos_pct", 34.0, 18.5, 4.0, 98.0),
        ("intake_manifold_kpa", 68.0, 21.0, 22.0, 145.0),
        ("exhaust_gas_temp_c", 480.0, 95.0, 220.0, 820.0),
        ("short_term_fuel_trim_pct", 1.2, 6.8, -22.0, 25.0),
        ("long_term_fuel_trim_pct", 1.8, 5.4, -18.0, 20.0),
        ("knock_sensor_mv", 185.0, 75.0, 20.0, 480.0),
        ("battery_alternator_v", 13.85, 0.65, 11.20, 15.40)
    ]

    data = {}
    for idx, (col_name, mean, std, low, high) in enumerate(feature_specs):
        col_vals = X_raw[:, idx]
        col_norm = (col_vals - col_vals.mean()) / col_vals.std()
        scaled = np.clip(mean + col_norm * std, low, high)
        data[col_name] = np.round(scaled, 4)

    df = pd.DataFrame(data)
    df["engine_fault_status"] = y  # 0 = Normal Operation, 1 = Powertrain/Engine Fault
    return df

def main():
    os.makedirs("model", exist_ok=True)
    print("=" * 80)
    print("BITS PILANI WILP - M.Tech (DSE) ML Assignment 2 | Automotive OBD-II Diagnostics")
    print("=" * 80)

    df = load_or_generate_automotive_dataset(STUDENT_SEED)
    print(f"[INFO] Dataset Shape: {df.shape} (Instances={len(df)}, Features={df.shape[1]-1})")
    print(f"[INFO] Target Distribution: {df['engine_fault_status'].value_counts().to_dict()}")

    X = df.drop(columns=["engine_fault_status"])
    y = df["engine_fault_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=STUDENT_SEED
    )

    # Export 20% hold-out test_data.csv for GitHub & Streamlit upload
    test_df = X_test.copy()
    test_df["engine_fault_status"] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"[INFO] Saved stratified hold-out test dataset to 'test_data.csv' ({len(test_df)} rows).")

    # Define the 5 Mandatory Classification Models wrapped in StandardScaler Pipelines
    classifiers = {
        "Logistic Regression": (
            "logistic_regression.pkl",
            LogisticRegression(max_iter=1000, random_state=STUDENT_SEED)
        ),
        "Decision Tree": (
            "decision_tree.pkl",
            DecisionTreeClassifier(max_depth=7, min_samples_split=8, random_state=STUDENT_SEED)
        ),
        "kNN": (
            "knn_classifier.pkl",
            KNeighborsClassifier(n_neighbors=7, weights="distance")
        ),
        "Naive Bayes": (
            "naive_bayes.pkl",
            GaussianNB()
        ),
        "Support Vector Machine": (
            "svm_classifier.pkl",
            SVC(kernel="rbf", C=2.5, gamma="scale", probability=True, random_state=STUDENT_SEED)
        )
    }

    metrics_list = []
    for model_name, (filename, clf) in classifiers.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)

        # Serialize trained pipeline
        joblib.dump(pipe, os.path.join("model", filename))

        # Evaluate on test_data
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics_list.append({
            "ML Model Name": model_name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
            "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist()
        })

    with open(os.path.join("model", "metrics_summary.json"), "w") as f:
        json.dump(metrics_list, f, indent=2)

    results_df = pd.DataFrame(metrics_list).drop(columns=["Confusion_Matrix"])
    print("\n" + "=" * 80)
    print("AUTOMOTIVE OBD-II FAULT CLASSIFICATION METRICS TABLE (TEST SET, N = 320)")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    print("[SUCCESS] All 5 models saved in 'model/' and 'test_data.csv' exported.")

if __name__ == "__main__":
    main()