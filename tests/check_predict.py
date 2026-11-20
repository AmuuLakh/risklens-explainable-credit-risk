import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from features import DROP_COLS
from predict import load_artifacts, logit, predict

art = load_artifacts()

df = pd.read_csv("data/processed/train_merged.csv")
sample = df.sample(500, random_state=1)

# 1) with the prediction function, giving plain dicts like an API would
applicants = sample.drop(columns=["TARGET", "SK_ID_CURR"]).to_dict(orient="records")
out = predict(applicants, art)
p_func = np.array([o["probability"] for o in out])

# 2) the notebook way (like in notebook 10), on the same rows
X = df.drop(columns=["TARGET", "SK_ID_CURR"] + DROP_COLS)
X = X.replace([np.inf, -np.inf], np.nan)
for c in X.select_dtypes(exclude="number").columns:
    X[c] = X[c].astype("category")
raw = art["model"].predict_proba(X.loc[sample.index])[:, 1]
p_nb = art["platt"].predict_proba(logit(raw).reshape(-1, 1))[:, 1]

diff = np.abs(p_func - p_nb).max()
print("biggest difference between the two ways:", diff)
print("threshold:", art["config"]["threshold"], "| flagged in the sample:", sum(o["flag"] for o in out), "of", len(out))
print("first 3 applicants:", out[:3])

assert diff < 1e-6, "the prediction function does not match the notebook way"
print("OK")