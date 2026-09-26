# Homework 1: Introduction to Machine Learning

Dataset: [Car Fuel Efficiency 2026](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)

## Questions

**Q1. Pandas version**
What version of Pandas did you install?

**Q2. Records count**
How many records are in the dataset?
- 5000 / 9000 / 10000 / 15000

**Q3. Fuel types**
How many fuel types are present in the dataset?
- 1 / 2 / 3 / 4

**Q4. Missing values**
How many columns have missing values?
- 0 / 1 / 2 / 3 / 4

**Q5. Max fuel efficiency**
What is the maximum fuel efficiency of cars from Asia?
- 21.2 / 31.2 / 41.2 / 51.2

**Q6. Median value of horsepower**
1. Find the median of `horsepower`
2. Find the most frequent value of `horsepower`
3. Fill missing values in `horsepower` with the most frequent value
4. Recalculate the median — has it changed?
- Yes, it increased / Yes, it decreased / No

**Q7. Sum of weights**
1. Select all cars from Asia
2. Select only `vehicle_weight` and `model_year`
3. Take the first 7 rows → matrix `X`
4. Compute `XTX = X.T @ X`
5. Invert `XTX`
6. Create `y = [1100, 1300, 800, 900, 1000, 1100, 1200]`
7. Compute `w = inv(XTX) @ X.T @ y`
8. What is the sum of all elements of `w`?
- 0.0369 / 0.369 / 3.69 / 36.9

## Run

```bash
cd homeworks-2026
uv run 01-intro/homework_01.py
```

## Submit

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01
