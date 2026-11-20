import json

import joblib
import numpy as np
import pandas as pd

from features import COUNT_COLS, add_basic_features

MODEL_DIR = "models/"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_artifacts(model_dir=MODEL_DIR):
    # the model, the calibrator, the threshold and the info about the columns
    with open(model_dir + "config.json") as f:
        config = json.load(f)
    with open(model_dir + "feature_info.json") as f:
        info = json.load(f)
    model = joblib.load(model_dir + config["model_file"])
    platt = joblib.load(model_dir + config["calibrator_file"])
    return {"model": model, "platt": platt, "config": config, "info": info}


def prepare(applicants, info):
    # applicants: a list of dicts, with the column names of the training data
    # columns that are missing are treated as missing values (NaN)
    df = pd.DataFrame(applicants)

    # the columns needed for the ratio features and the DAYS_EMPLOYED fix
    for col in ["DAYS_EMPLOYED", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_basic_features(df)

    # no bureau / previous history means zero credits, like in training
    for col in COUNT_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df.replace([np.inf, -np.inf], np.nan)

    # same columns in the same order as in training
    # (gender and age are not in this list, so they are ignored even if they are sent)
    df = df.reindex(columns=info["feature_names"])
    for col in df.columns:
        if col in info["categorical"]:
            # a value that was not in the training data becomes missing
            df[col] = pd.Categorical(df[col], categories=info["categorical"][col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def predict(applicants, artifacts):
    X = prepare(applicants, artifacts["info"])
    raw = artifacts["model"].predict_proba(X)[:, 1]
    p = artifacts["platt"].predict_proba(logit(raw).reshape(-1, 1))[:, 1]
    threshold = artifacts["config"]["threshold"]
    return [{"probability": float(x), "flag": bool(x >= threshold)} for x in p]