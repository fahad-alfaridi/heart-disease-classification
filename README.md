# Heart Disease Classification

Predicts whether a patient has heart disease from 13 clinical measurements, with an honest evaluation.

## Key finding: data leakage from duplicate rows
**70% of the rows in the dataset are exact duplicates** (723 of 1,026), leaving only 303 unique patients.
With a plain random split, **97% of the test rows have an identical copy in the training set**, which is how a simple KNN reaches ~90% accuracy. After deduplication, the same unscaled KNN drops to ~62%, so the original score was mostly memorisation.

## What this project does
- Detects and removes duplicates, and demonstrates the leakage with numbers.
- EDA: correlation with the target, numeric features and categorical rates by class.
- Leak-free modelling: stratified split, `Pipeline` (scaling + one-hot fitted on training folds only), K chosen by cross-validation.
- Compares KNN, Logistic Regression and Random Forest with repeated 5-fold CV (accuracy, recall, F1, ROC AUC).
- Final test evaluation (confusion matrix, ROC curve) and permutation feature importance.
- Fixes the "new patient" prediction so the patient is actually classified (with a probability).

## Results (cross-validated, 303 unique patients)
| Model | Accuracy | Recall | ROC AUC |
|---|---|---|---|
| KNN (K=5) | 0.83 | 0.88 | 0.89 |
| Logistic Regression | 0.84 | 0.88 | 0.90 |
| **Random Forest** | **0.85** | **0.90** | **0.92** |

Held-out test (61 patients): accuracy 0.79, ROC AUC 0.88. Most influential features: `thal`, `ca`, `oldpeak`.

## Limitations
Small dataset (303 patients), so estimates are noisy and the test set is small. No external validation. This is a learning project, not a medical tool.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook heart_disease_classification.ipynb
```
Data: `data/heart_updated.csv` (the original `heart.csv` plus one extra record; `data/heart.csv` is kept for reference).
