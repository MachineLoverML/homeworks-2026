import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

from utils import find_path

# Data loading & preparation
car_fuel_efficiency_2026 = find_path("car_fuel_efficiency_2026.csv")
df = pd.read_csv(car_fuel_efficiency_2026)
df = df.fillna(0)

target = "fuel_efficiency_mpg"

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.pop(target).values
y_val = df_val.pop(target).values
y_test = df_test.pop(target).values

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.to_dict(orient="records"))
X_val = dv.transform(df_val.to_dict(orient="records"))
X_test = dv.transform(df_test.to_dict(orient="records"))

feature_names = dv.get_feature_names_out()


# Q1: Decision stump

dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

split_feature_encoded = feature_names[dt.tree_.feature[0]]
split_feature_original = split_feature_encoded.split("=")[0]    # get the name from name=value

print(f"Q1: splitting feature = '{split_feature_original}'")


# Q2: Random forest RMSE

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

rmse = root_mean_squared_error(y_val, rf.predict(X_val))
print(f"Q2: RMSE (n_estimators=10) = {rmse:.4f}")


# Q3: Best n_estimators

print("Q3: Best n_estimators")
for n in [10, 50, 100, 150]:
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    rmse = root_mean_squared_error(y_val, rf.predict(X_val))
    print(f"  n_estimators={n:3d}: RMSE={rmse:.3f}")



# Q4: Best max_depth (mean RMSE across n_estimators)

print("Q4: Best max_depth")
for depth in [10, 15, 20, 25]:
    rmses = []
    for n in [10, 50, 100, 150]:
        rf = RandomForestRegressor(
            n_estimators=n, max_depth=depth, random_state=1, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rmses.append(root_mean_squared_error(y_val, rf.predict(X_val)))
    mean_rmse = np.mean(rmses)
    print(f"  max_depth={depth}: mean RMSE={mean_rmse:.3f}")



# Q5: Most important feature

rf = RandomForestRegressor(
    n_estimators=10, max_depth=20, random_state=1, n_jobs=-1
)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=feature_names)
candidates = ["vehicle_weight", "horsepower", "acceleration", "engine_displacement"]

candidate_importances = {}
for feat in feature_names:
    col = feat.split("=")[0]  # fmt of categorical feature is name=value
    if col in candidates:
        candidate_importances[col] = (
            candidate_importances.get(col, 0.0) + importances[feat]
        )

most_important = max(candidate_importances, key=candidate_importances.get)
print("Q5: feature importances:")
for k, v in sorted(candidate_importances.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")
print(f"most important = '{most_important}'")


# Q6: XGBoost – best eta

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(feature_names))
dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(feature_names))
watchlist = [(dtrain, "train"), (dval, "val")]

print("Q6: XGBoost eta comparison")
for eta in [0.3, 0.1]:
    evals_result = {}
    xgb_params = {
        "eta": eta,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "reg:squarederror",
        "nthread": 8,
        "seed": 1,
        "verbosity": 0,
    }
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=100,
        evals=watchlist,
        verbose_eval=False,
        evals_result=evals_result,
    )
    val_rmses = evals_result["val"]["rmse"]
    best_val_rmse = min(val_rmses)
    best_round = val_rmses.index(best_val_rmse) + 1
    print(f"  eta={eta}: best val RMSE={best_val_rmse:.4f} at round {best_round}")

