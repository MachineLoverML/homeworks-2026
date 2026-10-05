# Homework 2: Regression

Dataset: [Car Fuel Efficiency 2026](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)

## Questions

**EDA**
Does `fuel_efficiency_mpg` have a long tail?

**Q1. Missing values**
Which column has missing values?
- `engine_displacement` / `horsepower` / `vehicle_weight` / `model_year`

**Q2. Median horsepower**
What's the median of `horsepower`?
- 204 / 254 / 304 / 354

**Q3. Fill missing values**
Train a linear regression filling NAs with 0 vs. mean. Which gives better RMSE?
- With 0 / With mean / Both are equally good

**Q4. Regularization**
Try `r` values `[0, 0.01, 0.1, 1, 5, 10, 100]`. Which gives the best RMSE?
- 0 / 0.01 / 0.1 / 1 / 5 / 10 / 100

**Q5. Seed stability**
Try seeds `[0..9]`. What's the standard deviation of RMSE scores?
- 0.006 / 0.016 / 0.029 / 0.036

**Q6. Final model**
Train on train+val (seed 9, fill 0, r=0.001). What's the RMSE on the test set?
- 0.236 / 2.236 / 22.10 / 221.0

## Run

```bash
uv run 02-regression/homework_02.py
```

## Submit

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw02
