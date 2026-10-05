# Homework 4: Evaluation

Dataset: [Course Lead Scoring 2026](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv)

Target: `converted`

## Questions

**Q1. ROC AUC feature importance**
Which numerical variable has the highest AUC as a standalone predictor?
- `lead_score` / `number_of_courses_viewed` / `interaction_count` / `annual_income`

**Q2. Model AUC**
Train `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)` with `DictVectorizer`. What's the AUC on the validation set (rounded to 3 decimals)?
- 0.532 / 0.632 / 0.732 / 0.832

**Q3. Precision/Recall intersection**
Plot precision and recall across thresholds 0.0–1.0 (step 0.01). At which threshold do they intersect?
- 0.43 / 0.63 / 0.73 / 0.83

**Q4. F1 score**
At which threshold is F1 maximal?
- 0.21 / 0.41 / 0.61 / 0.81

**Q5. 5-Fold CV std**
Use `KFold(n_splits=5, shuffle=True, random_state=1)`. What's the std of AUC scores?
- 0.001 / 0.007 / 0.013 / 0.060

**Q6. Hyperparameter tuning**
Try `C` values `[0.000001, 0.001, 1]` with 5-fold CV. Which gives the best mean AUC?
- 0.000001 / 0.001 / 1

## Run

```bash
uv run 04-evaluation/homework_04.py
```

## Submit

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04
