import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

# so that the api can import the code in src/, whatever folder it is started from
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import load_artifacts, predict, prepare  # noqa: E402

TOP_N = 5  # how many reasons to return

app = FastAPI(
    title="RiskLens API",
    description=(
        "Estimates the probability that a loan applicant will default and explains the top reasons. "
        "Student project, not for real lending decisions. A flag means: look at this application more closely."
    ),
    version="1.0",
)

# loaded once when the api starts
art = load_artifacts(str(ROOT / "models") + "/")
explainer = shap.TreeExplainer(art["model"])
# the calibration is a straight line on the log-odds, so I scale the shap values by its slope
slope = float(art["platt"].coef_[0][0])


class Applicant(BaseModel):
    # the main fields are listed so they show up in the docs
    # any other column of the training data can be sent too, missing ones count as missing values
    # gender and age can be sent but the model ignores them
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "NAME_CONTRACT_TYPE": "Cash loans",
                "AMT_INCOME_TOTAL": 135000,
                "AMT_CREDIT": 450000,
                "AMT_ANNUITY": 25000,
                "AMT_GOODS_PRICE": 450000,
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.6,
                "EXT_SOURCE_3": 0.5,
                "DAYS_EMPLOYED": -2000,
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "OCCUPATION_TYPE": "Laborers",
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "bureau_count": 3,
                "prev_refused_rate": 0.0,
            }
        },
    )

    NAME_CONTRACT_TYPE: Optional[str] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None
    bureau_count: Optional[float] = None
    bureau_active: Optional[float] = None
    prev_count: Optional[float] = None
    prev_refused_rate: Optional[float] = None


class Reason(BaseModel):
    feature: str
    value: Optional[Any] = None
    effect: float
    direction: str


class Prediction(BaseModel):
    probability: float
    flag: bool
    threshold: float
    top_reasons: List[Reason]
    ignored_fields: List[str]
    unknown_category_values: List[str]
    n_features: int
    n_features_missing: int
    note: str


def explain(X, top_n=TOP_N):
    # shap values for one applicant, biggest effects first
    sv = explainer(X)
    values = sv.values
    if values.ndim == 3:  # some shap versions return one array per class
        values = values[:, :, 1]
    values = values[0] * slope

    reasons = []
    for i in np.argsort(-np.abs(values))[:top_n]:
        col = X.columns[i]
        v = X[col].iloc[0]
        if pd.isna(v):
            v = None
        elif not isinstance(v, str):
            v = float(v)
        reasons.append(
            Reason(
                feature=col,
                value=v,
                effect=round(float(values[i]), 3),
                direction="increases risk" if values[i] > 0 else "decreases risk",
            )
        )
    return reasons


@app.get("/health")
def health():
    return {
        "status": "ok",
        "threshold": art["config"]["threshold"],
        "n_features": len(art["info"]["feature_names"]),
    }


@app.post("/predict", response_model=Prediction)
def score_applicant(applicant: Applicant):
    data = applicant.model_dump(exclude_none=True)

    info = art["info"]
    feature_names = info["feature_names"]

    # at least one field the model really uses is needed
    if not any(k in feature_names for k in data):
        raise HTTPException(
            status_code=422,
            detail="Send at least one of the model's fields (see /docs for the main ones).",
        )

    result = predict([data], art)[0]
    X = prepare([data], info)

    return Prediction(
        probability=round(result["probability"], 4),
        flag=result["flag"],
        threshold=art["config"]["threshold"],
        top_reasons=explain(X),
        ignored_fields=[k for k in data if k not in feature_names],
        # a text value that was not in the training data is treated as missing
        unknown_category_values=[
            k for k, v in data.items()
            if k in info["categorical"] and v not in info["categorical"][k]
        ],
        n_features=len(feature_names),
        n_features_missing=int(X.isna().sum(axis=1).iloc[0]),
        note=(
            "Decision support only. The effects are in log-odds on the calibrated scale "
            "(positive = pushes the risk up), not percentage points. "
            "A value of null in the reasons means the field was missing. "
            "The more fields you send, the more reliable the result."
        ),
    )