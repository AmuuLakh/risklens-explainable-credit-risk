# RiskLens: Explainable Credit Risk

A credit default model that predicts whether a loan applicant is likely to default, and explains each prediction.

## Goal
Compare a transparent logistic regression scorecard with a gradient boosting model, then use SHAP to explain which factors drive risk.

## Data
Home Credit Default Risk (Kaggle). The data is not stored in this repo.
1. Join the competition on Kaggle and accept the rules.
2. Download the files and extract them into `data/raw/`.

## Project structure
- `notebooks/`: exploration and modeling
- `src/`: reusable code
- `data/`: local only, gitignored

## Status
Baseline and LightGBM done (ROC AUC 0.763). Next: SHAP explanations, fairness checks, API.

## Results

Validation set: 20% of the applicants (61,503 rows), stratified split, `random_state=42`. Threshold 0.5 for recall and precision.

| Model | ROC AUC | PR AUC | Recall | Precision |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.744 | 0.224 | 0.670 | 0.159 |
| LightGBM | 0.763 | 0.258 | 0.674 | 0.174 |

**Notes**
- 8.1% of the loans are defaults, so I didn't use accuracy. A random model would get a PR AUC of about 0.081.
- LightGBM improves on the baseline, but only modestly. At the same recall it makes about 1,600 fewer false alarms. I used mostly default settings and didn't tune hyperparameters.
- Most important features: the three `EXT_SOURCE` scores, then organization type, days employed and credit amount. Three features I built from the bureau and previous application tables are in the top 15.
- At a 0.5 threshold the model flags about 31% of applicants and only 17% of those flagged really default, so it's not ready for real decisions.
- `CODE_GENDER` is among the model's most used features, so fairness checks are the next step.