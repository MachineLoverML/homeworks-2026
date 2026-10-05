import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

def find_file(filename):
    if "__file__" in globals():
        return Path(__file__).parent.parent / filename
    
    path = Path.cwd()
    while not (path / filename).exists():
        path = path.parent
    return path / filename


car_fuel_efficiency_2026 = find_file("car_fuel_efficiency_2026.csv")
df = pd.read_csv(car_fuel_efficiency_2026)

features=[
'engine_displacement',
'horsepower',
'vehicle_weight',
'model_year',
]

target='fuel_efficiency_mpg'


df = df[features + [target]]
X = df[features]
y = df[target]


# Q0: EDA has fuel_efficiency_mpg a long tail?
sns.histplot(df[target], bins=50)
# plt.show() # show chart and pause the execution (close it to resume)
fuel_efficiency_mpg = car_fuel_efficiency_2026.parent \
        / "02-regression" \
        / "fuel_efficiency_mpg.png"
plt.savefig(fuel_efficiency_mpg)

print(f"Generated plot for fuel_efficiency_mpg: {fuel_efficiency_mpg}")
print("Q0: The data has already a bell shape. No need to take log1p.")
use_log1p = True
use_log1p = False
# Q1: what column has missing values
print("Q1: column with missing values: ", X.columns[X.isnull().any()][0])
# X.isnull().any()[X.isnull().any()]


# Q2 
print("Q2: median of horsepower: ", X['horsepower'].median())

# Q3
def split_dataset(df, seed):
    """Split dataset 60%/20%/20%."""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test
    
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)
    
    return df_train, df_val, df_test


def train_linear_regression_from_lesson(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

def rmse_from_lesson(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

def cp_df_after_filling_hp(df_orig, fill_value):
    df = df_orig.copy()
    df["horsepower"] = df["horsepower"].fillna(fill_value)
    return df

def train_and_calculate_error_from_lesson(df_train, df_val, fill_value, use_log1p):
    df = cp_df_after_filling_hp(df_train, fill_value)
    X_train = df[features].to_numpy()
    y_train = df[target].to_numpy()
    if use_log1p:
        y_train = np.log1p(y_train)
    
    w0, w = train_linear_regression_from_lesson(X_train, y_train)
    
    df = cp_df_after_filling_hp(df_val, fill_value)
    X_val = df[features].to_numpy()
    y_val = df[target].to_numpy()
    if use_log1p:
        y_val = np.log1p(y_val)
    
    y_pred = w0 + X_val.dot(w)
    score = rmse_from_lesson(y_val, y_pred)
    return round(score, 3)

df_train, df_val, df_test = split_dataset(df, 42)
mean = df_train["horsepower"].mean()
rmse0 = train_and_calculate_error_from_lesson(df_train, df_val, 0, use_log1p)
rmse_mean = train_and_calculate_error_from_lesson(df_train, df_val, mean, use_log1p)
print(f'Q3:\n  RMSE with 0 filling: {rmse0}\n  RMSE with mean filling: {rmse_mean}')
print('Both are equally good' if rmse0 == rmse_mean else 'One of the two is better')

def train_lr_with_reg(X, y, r):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

def train_with_reg(df_train, df_val, use_log1p, r_to_try):
    df = cp_df_after_filling_hp(df_train, 0)
    X_train = df[features].to_numpy()
    y_train = df[target].to_numpy()
    if use_log1p:
        y_train = np.log1p(y_train)
    
    df = cp_df_after_filling_hp(df_val, 0)
    X_val = df[features].to_numpy()
    y_val = df[target].to_numpy()
    if use_log1p:
        y_val = np.log1p(y_val)
    
    for r in r_to_try:
        w0, w = train_lr_with_reg(X_train, y_train, r)
        
        y_pred = w0 + X_val.dot(w)
        score = rmse_from_lesson(y_val, y_pred)
        print(f"RMSE(r={r}) = {round(score, 4)}")

r_to_try = [0, 0.01, 0.1, 1, 5, 10, 100]
train_with_reg(df_train, df_val, use_log1p, r_to_try)
print("Q4: best result with r=0")

rmse =[]
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    df_train, df_val, df_test = split_dataset(df, seed)
    rmse.append(
        train_and_calculate_error_from_lesson(df_train, df_val, 0, use_log1p))
std = np.std(rmse)
print(f"Q5: standard deviation of RMSE: {round(std, 3)}") # 0.029

seed = 9
r = 0.001
df_train, df_val, df_test = split_dataset(df, seed)

df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)

print("Q6: ", end="")
train_with_reg(df_full_train, df_test, use_log1p, [r])
