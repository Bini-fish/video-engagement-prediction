import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def engagement_model():
    """Train models to predict engagement probability for educational videos."""

    # Load data
    train_df = pd.read_csv("assets/train.csv", index_col="id")
    test_df = pd.read_csv("assets/test.csv", index_col="id")

    X_train = train_df.drop("engagement", axis=1)
    y_train = train_df["engagement"].astype(int)
    X_test = test_df.copy()

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models and grids
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=500, random_state=0),
            {"C": [0.01, 0.1, 1, 10]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=0),
            {"n_estimators": [50, 100, 200],
             "max_depth": [3, 5, 8, None],
             "max_features": ["sqrt", "log2", None]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=0),
            {"n_estimators": [100, 200],
             "learning_rate": [0.05, 0.1, 0.2],
             "max_depth": [2, 3, 4]}
        ),
    }

    best_model = None
    best_auc = 0
    best_name = None

    # Stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    model_auc_scores = {}

    for name, (model, param_grid) in models.items():
        grid = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1
        )
        grid.fit(X_train_scaled, y_train)

        mean_auc = grid.best_score_
        model_auc_scores[name] = mean_auc

        print(f"{name}: AUC = {mean_auc:.4f}, Best Params = {grid.best_params_}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model = grid.best_estimator_
            best_name = name

    print(f"\nBest model: {best_name} with AUC = {best_auc:.4f}")

    # Train on all data
    best_model.fit(X_train_scaled, y_train)

    # Predict probabilities
    predictions = best_model.predict_proba(X_test_scaled)[:, 1]
    result = pd.Series(predictions, index=test_df.index, name="engagement")

    # Save results
    result.to_csv("results/submission.csv")

    return result, model_auc_scores, best_model, X_train_scaled, y_train
