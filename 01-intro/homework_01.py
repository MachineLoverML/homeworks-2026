import pandas as pd
import numpy as np

# Q1
print("Q1 Pandas version:", pd.__version__)

df = pd.read_csv("01-intro/car_fuel_efficiency_2026.csv")

# Q2
print("Q2 Records:", len(df))

# Q3
print("Q3 Fuel types:", df["fuel_type"].nunique())

# Q4
print("Q4 Columns with missing values:", df.isnull().any().sum())

# Q5
asia_df = df[df["origin"] == "Asia"]
print("Q5 Max fuel efficiency (Asia):", asia_df["fuel_efficiency_mpg"].max())

# Q6
median_before = df["horsepower"].median()
mode_hp = df["horsepower"].mode()[0]
df["horsepower"] = df["horsepower"].fillna(mode_hp)
median_after = df["horsepower"].median()
print(f"Q6 Median before: {median_before} | after: {median_after} | changed: {median_before != median_after}")

# Q7
X = df[df["origin"] == "Asia"][["vehicle_weight", "model_year"]].head(7).values
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print("Q7 Sum of w:", w.sum())
