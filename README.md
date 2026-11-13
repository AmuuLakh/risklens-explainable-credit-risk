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
Baseline done (logistic regression, ROC AUC 0.744). Next: LightGBM, SHAP explanations, fairness checks, API.

## Results

Validation set: 20% of the applicants (61,503 rows), stratified split, `random_state=42`.

| Model | ROC AUC | PR AUC | Recall | Precision |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.744 | 0.224 | 0.670 | 0.159 |

**Baseline notes**
- 8.1% of the loans are defaults, so I didn't use accuracy. PR AUC of a random model would be about 0.081, so the baseline is roughly 2.8 times better.
- At a 0.5 threshold, the model catches 67% of defaulters but flags 34% of all applicants, so most flagged people are false alarms (precision 0.16). Choosing the threshold will matter for the final model.
- Only numeric columns are used so far. Text columns come with the next model.
- Some coefficients are hard to interpret because features are correlated (for example credit amount and goods price), which is one reason I'm adding SHAP.