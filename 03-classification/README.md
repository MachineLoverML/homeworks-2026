# Homework 3: Classification

Dataset: [Course Lead Scoring 2026](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv)

Target: `converted` (has the client signed up?)

## Questions

**Q1. Mode of `industry`**
What is the most frequent value in the `industry` column?
- `NA` / `technology` / `healthcare` / `retail`

**Q2. Correlation matrix**
Which two numerical features have the biggest correlation?
- `interaction_count` & `lead_score` / `number_of_courses_viewed` & `lead_score` / `number_of_courses_viewed` & `interaction_count` / `annual_income` & `interaction_count`

**Q3. Mutual information**
Which categorical variable has the highest mutual information score with `converted`?
- `industry` / `location` / `lead_source` / `employment_status`

**Q4. Logistic regression accuracy**
Train `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`. What's the accuracy on the validation set (rounded to 2 decimals)?
- 0.55 / 0.65 / 0.75 / 0.85

**Q5. Feature elimination**
Which feature has the smallest accuracy drop when excluded?
- `lead_source` / `number_of_courses_viewed` / `interaction_count`

**Q6. Regularization**
Try `C` values `[0.000001, 0.00001, 0.0001, 0.001]`. Which gives the best validation accuracy?
- 0.000001 / 0.00001 / 0.0001 / 0.001

## Run

```bash
uv run 03-classification/homework_03.py
```

## Submit

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03
