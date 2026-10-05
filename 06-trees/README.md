# Homework 6: Trees

Dataset: [Car Fuel Efficiency 2026](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)

Target: `fuel_efficiency_mpg`

## Questions

**Q1. Decision tree split**
Train a `DecisionTreeRegressor(max_depth=1)`. Which feature is used for the root split?
- `vehicle_weight` / `model_year` / `origin` / `fuel_type`

**Q2. Random forest RMSE**
Train `RandomForestRegressor(n_estimators=10, random_state=1)`. What's the RMSE on validation?
- 0.1837 / 1.837 / 18.37 / 183.7

**Q3. Best n_estimators**
Try `n_estimators` in `[10, 50, 100, 150]`. Which gives the lowest validation RMSE?
- 10 / 50 / 100 / 150

**Q4. Best max_depth**
Try `max_depth` in `[10, 15, 20, 25]` × `n_estimators` in `[10, 50, 100, 150]`. Which `max_depth` gives the best mean RMSE?
- 10 / 15 / 20 / 25

**Q5. Feature importance**
Train `RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1)`. What's the most important feature?
- `vehicle_weight` / `horsepower` / `acceleration` / `engine_displacement`

**Q6. XGBoost eta**
Train XGBoost for 100 rounds. Compare `eta=0.3` vs `eta=0.1`. Which gives the best validation RMSE?
- 0.3 / 0.1 / Both equal

## Run

```bash
uv run 06-trees/homework_06.py
```

## Submit

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw06
