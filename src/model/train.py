"""
Training script for the Proactive Guardian LightGBM classifier.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['risk_level', 'session_id']).values
    y = df['risk_level'].values - 1
    groups = df['session_id'].values
    return X, y, groups


def train_model(X_train, y_train, X_val, y_val):
    params = {
        'objective': 'multiclass',
        'num_class': 5,
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 400,
        'min_child_samples': 20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'class_weight': 'balanced',
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    return model


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    print("Classification Report:")
    print(classification_report(y_val, y_pred, digits=3))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))


if __name__ == "__main__":
    data_path = "data/processed/features.csv"

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Generate feature data first.")
    else:
        X, y, groups = load_data(data_path)
        unique_sessions = np.unique(groups)
        train_sessions, val_sessions = train_test_split(
            unique_sessions, test_size=0.3, random_state=42
        )
        train_mask = np.isin(groups, train_sessions)
        val_mask = np.isin(groups, val_sessions)

        model = train_model(X[train_mask], y[train_mask], X[val_mask], y[val_mask])
        evaluate(model, X[val_mask], y[val_mask])

        model.booster_.save_model("src/model/lightgbm_model.txt")
        print("Model saved to src/model/lightgbm_model.txt")
