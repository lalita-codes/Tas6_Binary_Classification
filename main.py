import yaml

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.train import train_model
from src.evaluate import (
    evaluate_model,
    evaluate_baseline,
    error_analysis,
    apply_threshold,
    confusion_matrix_analysis,
    roc_pr_analysis,
    threshold_optimization_curve,
    get_best_threshold
)
from src.utils import save_predictions
from src.evaluate import (
    save_report,
    save_confusion_matrix_plot,
    save_roc_curve,
    save_pr_curve,
    save_threshold_curve
)


# =========================
# Load config
# =========================
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    import os
    print("CONFIG PATH:", os.path.abspath("config/config.yaml"))
    print("CONFIG LOADED:", config["model"])


# =========================
# Load data
# =========================
df = load_data(config["data"]["raw_path"])


# =========================
# Clean data
# =========================
df = clean_data(df)


# =========================
# Train model
# =========================
target_column = "Class"

model, X_train, X_test, y_train, y_test = train_model(df, target_column,config)


# =========================
# Evaluate model
# =========================
results = evaluate_model(model, X_test, y_test)


# =========================
# Baseline evaluation
# =========================
baseline_results = evaluate_baseline(y_train, y_test)



# =========================
# Error analysis
# =========================
error_analysis(model, X_test, y_test)


# =========================
# THRESHOLD TESTING 
# =========================
print("\n========== THRESHOLD TESTING ==========")

y_prob = model.predict_proba(X_test)[:, 1]

for t in [0.2, 0.3, 0.4, 0.5]:
    y_pred = (y_prob >= t).astype(int)

    print("\nThreshold:", t)
    print("Predicted fraud cases:", sum(y_pred))


# Default prediction (0.3 threshold)
y_pred_default = apply_threshold(model, X_test, threshold=0.3)

# Confusion Matrix (business view)
confusion_matrix_analysis(y_test, y_pred_default)

# ROC + PR AUC
roc_pr_analysis(model, X_test, y_test)

# Threshold optimization graph
best_from_curve = threshold_optimization_curve(model, X_test, y_test)

# Auto best threshold selection
best_threshold = get_best_threshold(model, X_test, y_test)

# Final prediction using best threshold
final_predictions = apply_threshold(model, X_test, threshold=best_threshold)

# =========================
# SAVE PLOTS
# =========================
save_confusion_matrix_plot(model, X_test, y_test)
save_roc_curve(model, X_test, y_test)
save_pr_curve(model, X_test, y_test)
save_threshold_curve(model, X_test, y_test)


# =========================
# SAVE REPORT
# =========================
report = f"""
===========================
 FRAUD DETECTION REPORT
===========================

Model Metrics:
Accuracy: {results['accuracy']}
Precision: {results['precision']}
Recall: {results['recall']}
F1 Score: {results['f1_score']}

Baseline Metrics:
{baseline_results}

Best Threshold: {best_threshold}

Total Test Samples: {len(y_test)}
Fraud Detected: {sum(final_predictions)}
Actual Fraud: {sum(y_test)}

===========================
Generated Successfully
===========================
"""

save_report(report)


# =========================
# Save final predictions
# =========================
save_predictions(X_test, y_test, final_predictions)