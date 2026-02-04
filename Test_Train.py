from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import time
import numpy as np

def test_training_func(X_train, y_train, model_type, X_test, y_test, model_name, version, dataset_name):

    num_classes = len(np.unique(y_test))

    start_train = time.time()
    # use data in model
    model_type.fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True))
    end_train = time.time()

    start_pred = time.time()
    # then create predictions
    y_pred = model_type.predict(X_test.reset_index(drop=True))
    end_pred = time.time()
    print("model training time", round(end_train - start_train, 4))
    print("Model prediction time", round(end_pred - start_pred, 4))

    if hasattr(model_type, "predict_proba"): # for DT, kNN, NN
        y_probs = model_type.predict_proba(X_test.reset_index(drop=True))
    elif hasattr(model_type, "decision_function"): # for SVM
        y_probs = model_type.decision_function(X_test.reset_index(drop=True))
    else:
       raise ValueError("No predict_proba or decision_function present.")


    # accuracy check
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # ROC-AUC for data analysis

    if dataset_name.lower() == "accidents" and num_classes > 2:
        roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
        pr_auc = average_precision_score(y_test, y_probs, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        roc_auc = roc_auc_score(y_test, y_probs[:, 1])
        pr_auc = average_precision_score(y_test, y_probs[:, 1])
        f1 = f1_score(y_test, y_pred)

    # roc_auc = roc_auc_score(y_test, y_probs)
    print(f"ROC-AUC results: {roc_auc:.4f}")

    # PR-AUCC for data analysis
    # pr_auc = average_precision_score(y_test, y_probs)
    print(f"PR-AUC results: {pr_auc:.4f}")

    prevalence = 0
    # Prevalence for data analysis
    if num_classes == 2:
        prevalence = y_test.mean()
        print(f"Positive Class Prevalence (Baseline for PR-AUC): {prevalence:.4f}")
    else:
        print("no prevalence metric for multiclass classification")
    
    #prevalence = y_test.mean()
    print(f"Positive Class Prevalence Metrics (Baseline for PR-AUC): {prevalence:.4f}")

    # F1 Score analysis
    # f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")

    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save key metrics to a text file
    report_filename = f"Eval_Report_for_{dataset_name}_{model_name}_{version}.txt"
    with open(report_filename, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        if prevalence is not None:
            f.write(f"Positive Class Prevalence (Baseline for PR-AUC): {prevalence:.4f}\n")
        else:
            f.write("Positive Class Prevalence: N/A (Multiclass setting)\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix: {model_name}")

    # Save the plot instead of showing it
    save_name = model_name.replace(" ", "_")
    filename = f"Conf_Matrix_for_{dataset_name}_{save_name}_{version}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix as: {filename}")

    return acc, roc_auc, pr_auc, prevalence, f1

def run_model(model, X_train, y_train, X_test, y_test, model_name, version, dataset_name, X_sample):
    return test_training_func(X_train, y_train, model, X_test, y_test, model_name, version, dataset_name)

def run_model_set(model_list, dataset_name, X_train, X_test, y_train, y_test, pipeline_func_scaled, pipeline_func_unscaled, X_sample):
    results = []

    for item in model_list:
        model = item["model"]
        version = item["version"]
        name = item["name"]
        use_scaled = item["scaled"]

        pipeline = pipeline_func_scaled(model, X_sample) if use_scaled else pipeline_func_unscaled(model, X_sample)

        print(f"[{dataset_name}] Running {name} model with {'scaled' if use_scaled else 'unscaled'} data")
        # pipeline.fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True))
        result = run_model(pipeline, X_train, y_train, X_test, y_test, name, version, dataset_name, X_sample)
        results.append((dataset_name, name, version, result))

    print(results)

def us_accidents_stratified_split(us_accidents):
    #for SVM linear and DT
    cut1_df, _ = train_test_split(
    us_accidents,
    train_size=1_000_000,
    stratify=us_accidents['Severity'],
    random_state=42
    )

    # SVM kernel
    cut2_df, _ = train_test_split(
    us_accidents,
    train_size=100_000,
    stratify=us_accidents['Severity'],
    random_state=42
    )

    # 250k training + 25k testing (kNN exact model)
    cut3_temp_df, _ = train_test_split(
    us_accidents,
    train_size=275_000,
    stratify=us_accidents['Severity'],
    random_state=42
    )
    # and then split cut 3 for kNN
    cut3_train_df, cut3_test_df = train_test_split(
    cut3_temp_df,
    train_size=250_000,
    stratify=cut3_temp_df['Severity'],
    random_state=42
    )

    # for NN (80% of dataset)
    cut4_nn_df, _ = train_test_split(
    us_accidents,
    train_size=0.8,
    stratify=us_accidents['Severity'],
    random_state=42
    )

def plot_learning_curve(model, X, y, model_name, version, cv=5, scoring='accuracy'):
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("X or y is empty — check preprocessing requirements")
    

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 0.8, 5), cv=cv,
        scoring=scoring, n_jobs=-1, shuffle=True, random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label='Training Score')
    plt.plot(train_sizes, val_mean, label='Validation Score')
    plt.title(f"Learning Curve: {model_name}_{version}")
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Learning_Curve_{model_name.replace(' ', '_')}_{version}.png")
    plt.close()

def plot_validation_curve(model, X, y, param_name, param_range, model_name, cv=5, scoring='accuracy'):
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
    plt.plot(param_range, np.mean(val_scores, axis=1), label='Validation Score')
    plt.title(f"Model Complexity Curve: {model_name} {version} ({param_name})")
    plt.xlabel(param_name)
    plt.ylabel(scoring.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Validation_Curve_{model_name.replace(' ', '_')}_{version}.png")
    plt.close()

def plot_roc_pr_calibration(y_true, y_probs, model_name, version):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ROC_Curve_{model_name.replace(' ', '_')}_{version}.png")
    plt.close()

    # PR
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label="PR Curve")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"PR_Curve_{model_name.replace(' ', '_')}_{version}.png")
    plt.close()

    # Calibration
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.title("Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Calibration_Curve_{model_name.replace(' ', '_')}_{version}.png")
    plt.close()


