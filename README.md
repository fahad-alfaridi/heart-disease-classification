# Heart Disease Classification

Predicts whether a patient has heart disease from 13 clinical measurements, with an honest evaluation.

## Two data problems found and fixed
1. **70% of the rows are exact duplicates** (723 of 1,026), leaving only 303 unique patients. With a plain random split, **97% of the test rows have an identical copy in the training set**, which is how a simple KNN reaches ~90% accuracy. After deduplication the same unscaled KNN drops to ~62%, so the original score was mostly memorisation.
2. **The `target` column is reversed** in this Kaggle/UCI file: `target = 1` means *no* disease. Evidence (all in the notebook): the `target = 1` group has a higher maximum heart rate (158 vs 139), less exercise-induced angina (14% vs 55%), less ST depression, fewer narrowed vessels and fewer men, and its size (165) matches the 164 disease-free patients of the original UCI Cleveland data. The label is flipped (`disease = 1 - target`) so that sensitivity, confusion matrices and probabilities refer to sick patients.

## What this project does
- Detects duplicates and the reversed label, and demonstrates the leakage with numbers.
- EDA: correlations and feature distributions by disease status.
- Leak-free modelling: stratified split, `Pipeline` (scaling + one-hot fitted on training folds only), K chosen by cross-validation.
- Compares KNN, Logistic Regression and Random Forest with repeated 5-fold CV (accuracy, sensitivity, specificity, F1, ROC AUC) and a majority-class baseline.
- Final test evaluation, a **10-split robustness check**, and permutation feature importance averaged over those splits.
- Fixes the "new patient" prediction so the patient is actually classified (with a probability of disease).

## Results (303 unique patients, positive class = disease, majority baseline accuracy 54.5%)
Cross-validated on the training set (repeated 5-fold):

| Model | Accuracy | Sensitivity | Specificity | ROC AUC |
|---|---|---|---|---|
| KNN (K=14) | 0.83 | 0.71 | 0.93 | 0.89 |
| Logistic Regression | 0.84 | 0.79 | 0.88 | 0.91 |
| Random Forest | 0.84 | 0.79 | 0.89 | 0.92 |

**Robustness:** one split of 61 patients is very noisy, so the notebook repeats the stratified split 10 times. Test ROC AUC: Logistic Regression 0.906 (±0.022), Random Forest 0.903 (±0.032), KNN 0.885 (±0.026); test accuracy about 0.83 for both Logistic Regression and Random Forest (range 0.77 to 0.87). Logistic Regression and Random Forest are practically equivalent, so the ranking between them should not be over-interpreted. Most influential features (averaged over the splits): `thal`, `ca`, `cp`.

## Limitations
Small dataset (303 patients), so estimates are noisy and a single test split can move accuracy by several points. Sensitivity of about 0.79 means roughly one in five sick patients is missed. No external validation. This is a learning project, not a medical tool.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook heart_disease_classification.ipynb
```
Data: `data/heart_updated.csv` (the original `heart.csv` plus one extra record; `data/heart.csv` is kept for reference).
