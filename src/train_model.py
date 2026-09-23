import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def run_training():
    print("--- Starting Model Training and Evaluation ---")

    # 1. Load data/processed/train.csv and test.csv, and the feature column list
    print("Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    with open('data/processed/feature_columns.txt', 'r') as f:
        feature_columns = [line.strip() for line in f if line.strip()]

    X_train = train_df[feature_columns]
    y_train = train_df['is_malicious']
    X_test = test_df[feature_columns]
    y_test = test_df['is_malicious']

    print(f"Train data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    # 2. Print the class balance of y_train before SMOTE
    print("\n--- Class balance of y_train (before SMOTE) ---")
    print(y_train.value_counts())

    # 3. Apply SMOTE to the training set only
    print("\nApplying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("Class balance of y_train (after SMOTE):")
    print(y_train_smote.value_counts())

    # 4. Train a scikit-learn LogisticRegression
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1) # n_jobs for parallel processing
    model.fit(X_train_smote, y_train_smote)
    print("Model training complete.")

    # 5. Evaluate on the untouched, original test set
    print("\n--- Evaluating Model on Test Set ---")
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # 6. Specifically call out and print the F1-score, precision, and recall for the malicious class (class "1")
    print("\n--- Metrics for Malicious Class (1) ---")
    malicious_precision = class_report['1']['precision']
    malicious_recall = class_report['1']['recall']
    malicious_f1_score = class_report['1']['f1-score']
    print(f"Precision (Malicious): {malicious_precision:.4f}")
    print(f"Recall (Malicious):    {malicious_recall:.4f}")
    print(f"F1-score (Malicious):  {malicious_f1_score:.4f}")

    # 7. Save the trained model and evaluation results
    print("\nSaving model and evaluation results...")
    joblib.dump(model, 'models/baseline_logreg.joblib')
    print("Trained model saved to models/baseline_logreg.joblib")

    # Convert numpy array to list for JSON serialization
    conf_matrix_list = conf_matrix.tolist()

    evaluation_results = {
        'classification_report': class_report,
        'confusion_matrix': conf_matrix_list
    }
    with open('models/training_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print("Evaluation results saved to models/training_results.json")

    print("--- Model Training and Evaluation Complete ---")

if __name__ == '__main__':
    run_training()
