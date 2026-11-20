import json
import os

import joblib
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from features import DROP_COLS, build_table, make_xy

MODEL_DIR = "models/"

# my assumption: a missed default costs 5 times as much as a false alarm
# so the threshold is 1 / (1 + 5) = 0.167 (see notebook 10)
COST_RATIO = 5


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    print("building the table from the raw files...")
    df = build_table()
    X, y, cat_cols = make_xy(df)
    print("features:", X.shape, "| text columns:", len(cat_cols))

    # same splits as in the notebooks (random_state=42 everywhere)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    idx_cal, idx_test = train_test_split(
        np.arange(len(y_val)), test_size=0.5, stratify=y_val, random_state=42
    )

    # model (same settings as notebook 06)
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_es, y_es)],
        eval_metric="auc",
        callbacks=[early_stopping(100), log_evaluation(100)],
    )
    print("best number of trees:", model.best_iteration_)

    # calibration on one half of the validation set (notebook 08)
    raw = model.predict_proba(X_val)[:, 1]
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(logit(raw[idx_cal]).reshape(-1, 1), y_val.iloc[idx_cal].values)

    # check on the other half, which the calibrator never saw
    p_test = platt.predict_proba(logit(raw[idx_test]).reshape(-1, 1))[:, 1]
    y_test = y_val.iloc[idx_test].values
    print("test ROC AUC:", round(roc_auc_score(y_test, p_test), 4), "(notebook 10: 0.7647)")
    print("test Brier:", round(brier_score_loss(y_test, p_test), 4), "(notebook 10: 0.0673)")

    # save everything the prediction code needs
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_DIR + "lgbm_model_nosens.joblib")
    joblib.dump(platt, MODEL_DIR + "platt_calibrator_nosens.joblib")

    # the order of the columns and the categories of the text columns
    # are needed so that new applicants are prepared in exactly the same way
    feature_info = {
        "feature_names": list(X.columns),
        "categorical": {c: [str(v) for v in X[c].cat.categories] for c in cat_cols},
    }
    with open(MODEL_DIR + "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    config = {
        "threshold": round(1 / (1 + COST_RATIO), 4),
        "assumed_cost_ratio": COST_RATIO,
        "dropped_columns": DROP_COLS,
        "model_file": "lgbm_model_nosens.joblib",
        "calibrator_file": "platt_calibrator_nosens.joblib",
    }
    with open(MODEL_DIR + "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("saved the model, the calibrator, feature_info.json and config.json in", MODEL_DIR)


if __name__ == "__main__":
    main()