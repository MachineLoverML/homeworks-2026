import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from utils import find_path

course_lead_scoring_2026 = find_path("course_lead_scoring_2026.csv")
df = pd.read_csv(course_lead_scoring_2026)

categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']
target = 'converted'

for col in categorical:
    df[col] = df[col].fillna('NA')
for col in numerical:
    df[col] = df[col].fillna(0.0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Q1: ROC AUC per numerical feature as standalone predictor
print("Q1: AUC per numerical feature")
for col in numerical:
    auc = roc_auc_score(df_train[target], df_train[col])
    if auc < 0.5:
        auc = roc_auc_score(df_train[target], -df_train[col])
    print(f"  {col}: {round(auc, 3)}")

# Q2: Model AUC on validation set
y_train = df_train[target].to_numpy()
y_val = df_val[target].to_numpy()

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train[categorical + numerical].to_dict(orient='records'))
X_val = dv.transform(df_val[categorical + numerical].to_dict(orient='records'))

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_pred)
print(f"Q2: Validation AUC = {round(auc_val, 3)}")

# Q3 & Q4: Precision/recall/F1 across thresholds
thresholds = np.arange(0.0, 1.01, 0.01)
precisions, recalls, f1s = [], [], []

for t in thresholds:
    pred = (y_pred >= t).astype(int)
    tp = ((pred == 1) & (y_val == 1)).sum()
    fp = ((pred == 1) & (y_val == 0)).sum()
    fn = ((pred == 0) & (y_val == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)

precisions = np.array(precisions)
recalls = np.array(recalls)
f1s = np.array(f1s)

# Q3: intersection = smallest |precision - recall| ignoring both-zero cases
both_nonzero = (precisions > 0) | (recalls > 0)
diff = np.abs(precisions - recalls)
diff[~both_nonzero] = np.inf
intersection_idx = np.argmin(diff)
print(f"Q3: Precision/Recall intersection at threshold = {round(thresholds[intersection_idx], 2)}")

# Q4: threshold with max F1
best_f1_idx = np.argmax(f1s)
print(f"Q4: Max F1 = {round(f1s[best_f1_idx], 3)} at threshold = {round(thresholds[best_f1_idx], 2)}")

# Q5: 5-Fold CV std
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
aucs = []
for train_idx, val_idx in kfold.split(df_full_train):
    dv_k = DictVectorizer(sparse=False)
    X_k_train = dv_k.fit_transform(
        df_full_train.iloc[train_idx][categorical + numerical].to_dict(orient='records')
    )
    X_k_val = dv_k.transform(
        df_full_train.iloc[val_idx][categorical + numerical].to_dict(orient='records')
    )
    y_k_train = df_full_train.iloc[train_idx][target].to_numpy()
    y_k_val = df_full_train.iloc[val_idx][target].to_numpy()
    m = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    m.fit(X_k_train, y_k_train)
    aucs.append(roc_auc_score(y_k_val, m.predict_proba(X_k_val)[:, 1]))

print(f"Q5: 5-Fold CV AUC scores: {[round(a, 3) for a in aucs]}")
print(f"Q5: std = {round(np.std(aucs), 3)}")

# Q6: Hyperparameter tuning C values
print("Q6: Mean AUC per C value")
for C in [0.000001, 0.001, 1]:
    aucs_c = []
    for train_idx, val_idx in kfold.split(df_full_train):
        dv_c = DictVectorizer(sparse=False)
        X_c_train = dv_c.fit_transform(
            df_full_train.iloc[train_idx][categorical + numerical].to_dict(orient='records')
        )
        X_c_val = dv_c.transform(
            df_full_train.iloc[val_idx][categorical + numerical].to_dict(orient='records')
        )
        y_c_train = df_full_train.iloc[train_idx][target].to_numpy()
        y_c_val = df_full_train.iloc[val_idx][target].to_numpy()
        m = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        m.fit(X_c_train, y_c_train)
        aucs_c.append(roc_auc_score(y_c_val, m.predict_proba(X_c_val)[:, 1]))
    print(f"  C={C}: mean AUC = {round(np.mean(aucs_c), 3)}, std = {round(np.std(aucs_c), 3)}")
