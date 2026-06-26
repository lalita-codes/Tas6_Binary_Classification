from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# MAIN EVALUATION FUNCTION
# =========================
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    print("\nModel Performance:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results


# =========================
# BASELINE MODEL
# =========================
def baseline_model(y_train, y_test):

    majority_class = pd.Series(y_train).mode()[0]
    baseline_pred = [majority_class] * len(y_test)

    return baseline_pred


def evaluate_baseline(y_train, y_test):

    y_pred = baseline_model(y_train, y_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    print("\nBaseline Performance:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results


# =========================
# THRESHOLD FUNCTION
# =========================
def apply_threshold(model, X_test, threshold=0.3):

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return y_pred


# =========================
# ERROR ANALYSIS
# =========================
def error_analysis(model, X_test, y_test):

    y_pred = model.predict(X_test)

    df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    wrong = df[df["Actual"] != df["Predicted"]]

    print("\nError Analysis:")
    print("Total errors:", len(wrong))
    print(wrong.head())

    return wrong


# ==========================================================
# CONFUSION MATRIX 
# ==========================================================
def confusion_matrix_analysis(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)

    print("\nBusiness Interpretation:")
    print(f"True Negatives: {tn}")
    print(f"False Positives (false alarms): {fp}")
    print(f"False Negatives (missed fraud): {fn}")
    print(f"True Positives: {tp}")

    return cm


# ==========================================================
# ROC + PR AUC
# ==========================================================
def roc_pr_analysis(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    print("\nROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    return roc_auc, pr_auc


# ==========================================================
#  THRESHOLD OPTIMIZATION CURVE
# ==========================================================
def threshold_optimization_curve(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold Optimization Curve")
    plt.show()

    best_threshold = thresholds[np.argmax(f1_scores)]

    print("\nBest Threshold:", best_threshold)

    return best_threshold


# ==========================================================
# AUTO BEST THRESHOLD
# ==========================================================
def get_best_threshold(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]

    best_t = 0.5
    best_f1 = 0

    for t in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_test, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_t = t

    print("\nBest Threshold Found:")
    print("Threshold:", best_t)
    print("F1 Score:", best_f1)

    return best_t

# ==========================================================
# Save the Report function
# ==========================================================

def save_report(text, filename="outputs/model_report.txt"):
    os.makedirs("outputs", exist_ok=True)

    with open(filename, "w") as f:
        f.write(text)

    print(f"\nReport saved at: {filename}")

# ==========================================================
# Save ConfusionMatrix Plot
# ==========================================================
from sklearn.metrics import ConfusionMatrixDisplay

def save_confusion_matrix_plot(model, X_test, y_test):
    os.makedirs("outputs/plots", exist_ok=True)

    y_pred = model.predict(X_test)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")

    path = "outputs/plots/logistic_confusion_matrix.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")

# ==========================================================
# Save ROC Curve Plot
# ==========================================================
from sklearn.metrics import roc_curve

def save_roc_curve(model, X_test, y_test):
    os.makedirs("outputs/plots", exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    path = "outputs/plots/logistic_roc_curve.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")

# ==========================================================
# Save Precision-Recall Curve Plot
# ==========================================================
def save_pr_curve(model, X_test, y_test):

    os.makedirs("outputs/plots", exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    path = "outputs/plots/logistic_pr_curve.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")

# ==========================================================
# Save Threshold Curve Plot
# ==========================================================
def save_threshold_curve(model, X_test, y_test):

    os.makedirs("outputs/plots", exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold Optimization Curve")

    path = "outputs/plots/logistic_threshold_curve.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")