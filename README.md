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

- **The scores are not real probabilities.** I trained with `class_weight="balanced"` because only 8% of the loans are defaults, so the model scores as if half of the applicants were defaults. The ranking (AUC) and the SHAP effects are still meaningful, but a score of 0.9 does not mean a 90% chance of default. Calibration is the next step.
- SHAP shows what the model uses, not what causes default.
- No hyperparameter tuning, and a single random split (there is no date column in the data).
- Only 5,000 rows are used for the SHAP plots.
- The model currently uses gender and age. I'm checking what this does in the fairness part.