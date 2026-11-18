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
Baseline, LightGBM, SHAP and calibration done. Next: fairness checks, API, Docker, drift monitoring.

## Results

Validation set: 20% of the applicants (61,503 rows), stratified split, `random_state=42`. Threshold 0.5 for recall and precision.

| Model | ROC AUC | PR AUC | Recall | Precision |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.744 | 0.224 | 0.670 | 0.159 |
| LightGBM | 0.763 | 0.258 | 0.674 | 0.174 |


### Calibration

I trained with balanced class weights, so the raw scores were too high (average 0.40 against a real default rate of 0.08). I calibrated them with Platt scaling, fitted on half of the validation set and tested on the other half.

| Scores | Mean predicted | Brier | Log loss | ROC AUC |
|---|---|---|---|---|
| Raw | 0.399 | 0.186 | 0.552 | 0.766 |
| Platt-calibrated | 0.080 | 0.067 | 0.244 | 0.766 |

The real default rate is 0.081, and always predicting 8% gives a Brier score of 0.074.

![Calibration curve](reports/figures/calibration_curve.png)

After calibration, the real default rate goes from **1.3%** in the safest 10% of applicants to **28.6%** in the riskiest 10%, and the predicted probabilities match these rates closely. The riskiest 10% contain roughly a third of all defaults. Even so, about 7 in 10 applicants in the riskiest group repay, so the model sorts applicants by risk but can't say who will default.


**Notes**
- 8.1% of the loans are defaults, so I didn't use accuracy. A random model would get a PR AUC of about 0.081.
- LightGBM improves on the baseline, but only modestly. At the same recall it makes about 1,600 fewer false alarms. I used mostly default settings and didn't tune hyperparameters.
- Most important features: the three `EXT_SOURCE` scores, then organization type, days employed and credit amount. Three features I built from the bureau and previous application tables are in the top 15.
- At a 0.5 threshold the model flags about 31% of applicants and only 17% of those flagged really default, so it's not ready for real decisions.
- `CODE_GENDER` is among the model's most used features, so fairness checks are the next step.


## Explaining the model (SHAP)

I used SHAP on the LightGBM model to see why it gives a high or low risk score. The plots use a random sample of 5,000 validation applicants.

![SHAP bar plot](reports/figures/shap_bar.png)
![SHAP beeswarm plot](reports/figures/shap_beeswarm.png)

**What I found**
- The three external scores (`EXT_SOURCE_1/2/3`) matter far more than anything else. Low scores push the risk up strongly, and the effect of `EXT_SOURCE_3` is almost a straight line.
- A short time in the current job and a high share of refused previous applications also push the risk up.
- Three features I built from the bureau and previous application tables are in the top 15.
- Gender is the 6th most important feature for the model, and age is in the model twice (`DAYS_BIRTH` and `AGE`), so checking fairness is my next step.

Two single applicants (the highest and lowest risk in the sample):

![Waterfall, high risk](reports/figures/shap_waterfall_high.png)
![Waterfall, low risk](reports/figures/shap_waterfall_low.png)

The dependence plots for `EXT_SOURCE_3` and `DAYS_EMPLOYED` are in `reports/figures/`.

## Limitations

-**Calibration only holds for this dataset.** I calibrated the scores with Platt scaling on half of the validation set. The probabilities match the real default rate here, but if the applicants or the default rate changed, I'd have to calibrate again.
- Recall and precision in the results table use a 0.5 threshold on the *raw* scores, before calibration. I'll choose a threshold on the calibrated probabilities later.
- The SHAP plots are on the raw (uncalibrated) scale. Platt scaling is a straight line on the log-odds, so the directions and rankings of the features don't change, only the size of the effects.
- SHAP shows what the model uses, not what causes default.
- No hyperparameter tuning, and a single random split (there is no date column in the data).
- Only 5,000 rows are used for the SHAP plots.
- The model currently uses gender and age. I'm checking what this does in the fairness part.