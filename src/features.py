import numpy as np
import pandas as pd

RAW_DIR = "data/raw/home-credit-default-risk/"

# gender and age are not used by the final model (see the fairness notebook)
DROP_COLS = ["CODE_GENDER", "DAYS_BIRTH", "AGE"]

# for people with no bureau / previous history these counts are 0, not missing
COUNT_COLS = ["bureau_count", "bureau_active", "prev_count", "prev_refused"]


def bureau_summary(raw_dir=RAW_DIR):
    # one row per applicant, from the bureau table (notebook 02)
    bureau = pd.read_csv(raw_dir + "bureau.csv")
    bureau["ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    return bureau.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        bureau_active=("ACTIVE", "sum"),
        bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
        bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_max_overdue=("CREDIT_DAY_OVERDUE", "max"),
    ).reset_index()


def previous_summary(raw_dir=RAW_DIR):
    # one row per applicant, from the previous applications table (notebook 03)
    prev = pd.read_csv(raw_dir + "previous_application.csv")
    prev["REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    return prev.groupby("SK_ID_CURR").agg(
        prev_count=("SK_ID_PREV", "count"),
        prev_refused=("REFUSED", "sum"),
        prev_refused_rate=("REFUSED", "mean"),
        prev_avg_credit=("AMT_CREDIT", "mean"),
    ).reset_index()


def add_basic_features(df):
    # the fixes and ratio features from notebook 04
    # used for training AND for prediction, so both do exactly the same thing
    df = df.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    return df


def build_table(raw_dir=RAW_DIR):
    # the whole merged table, from the raw csv files
    app = pd.read_csv(raw_dir + "application_train.csv")
    df = app.merge(bureau_summary(raw_dir), on="SK_ID_CURR", how="left")
    df = df.merge(previous_summary(raw_dir), on="SK_ID_CURR", how="left")
    df = add_basic_features(df)
    df[COUNT_COLS] = df[COUNT_COLS].fillna(0)
    return df


def make_xy(df):
    # features and target for the model, without the sensitive columns
    y = df["TARGET"]
    drop = ["TARGET", "SK_ID_CURR"] + [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)
    X = X.replace([np.inf, -np.inf], np.nan)

    # text columns become categories, lightgbm can use them directly
    cat_cols = list(X.select_dtypes(exclude="number").columns)
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, y, cat_cols