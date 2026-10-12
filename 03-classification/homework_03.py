import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from utils import find_path

course_lead_scoring_2026 = find_path("course_lead_scoring_2026.csv")
df = pd.read_csv(course_lead_scoring_2026)

# Data Preparation: replace with 'NA' for categorical features, 0.0 for numbers
df.info()
# RangeIndex: 5000 entries, 0 to 4999
# Data columns (total 9 columns):
#  #   Column                    Non-Null Count  Dtype  
# ---  ------                    --------------  -----  
#  0   lead_source               4852 non-null   str    
#  1   industry                  4760 non-null   str    
#  2   employment_status         4806 non-null   str    
#  3   location                  4792 non-null   str    
#  4   annual_income             4631 non-null   float64
#  5   number_of_courses_viewed  5000 non-null   int64  
#  6   interaction_count         5000 non-null   int64  
#  7   lead_score                4965 non-null   float64
#  8   converted                 5000 non-null   int64  
# dtypes: float64(2), int64(3), str(4)
categorical = [
'lead_source',
'industry',
'employment_status',
'location',
]
numerical=[
'annual_income',
'number_of_courses_viewed',
'interaction_count',
'lead_score',
]
target = 'converted'

df[ df[categorical].isnull() ] = 'NA'
df[ df[numerical].isnull() ] = 0.0

# Q1 
print(f"Q1: most frequent value in 'industry': {df['industry'].mode()[0]}")

# Q2
pair = [
  [
    'interaction_count',
    'lead_score',
  ],[
    'number_of_courses_viewed',
    'lead_score',
  ],[
    'number_of_courses_viewed',
    'interaction_count',
  ],[
    'annual_income',
    'interaction_count',
  ]
]
for i in range(len(pair)):
  print(f"Corr({pair[i][0]},{pair[i][1]}) = {df[pair[i][0]].corr(df[pair[i][1]])}")

print("Q2: biggest Corr(interaction_count,lead_score) = 0.916")

# Split the data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# Q3
score = []
for cat in categorical:
  score.append(mutual_info_score(df_train[target], df_train[cat]))

for i in zip(categorical, score):
  print(f"{i[0]}: {round(i[1],2)}")

print("Q3: biggest mutual_info: lead_source")

# Q4
y_train = df_train[target].to_numpy()
df_train = df_train.drop(columns=[target])

y_val = df_val[target].to_numpy()
df_val = df_val.drop(columns=[target])

def train_and_evaluate(df_train, y_train, df_val, y_val, col_to_drop=None, C=1.0):
  if col_to_drop:
      df_train = df_train.copy()
      df_train = df_train.drop(columns=[col_to_drop])
      df_val = df_val.copy()
      df_val = df_val.drop(columns=[col_to_drop])
  
  dv = DictVectorizer(sparse=False)
  
  X_train = dv.fit_transform(df_train.to_dict(orient="records"))
  X_val = dv.transform(df_val.to_dict(orient="records"))
  
  model = LogisticRegression(
    solver='liblinear', C=C, max_iter=1000, random_state=42)
  model.fit(X_train, y_train)
  
  accuracy = (model.predict(X_val) == y_val).mean()
  return accuracy

Q4_accuracy = train_and_evaluate(df_train, y_train, df_val, y_val)
print(f"Q4 accuracy of LogisticRegression: {round(Q4_accuracy,2)}")

# Q5
for cat in ["lead_source", "number_of_courses_viewed", "interaction_count"]:
  acc = train_and_evaluate(df_train, y_train, df_val, y_val, cat)
  print(f"Acc without {cat}: Q4_acc({Q4_accuracy}) - {acc} = {Q4_accuracy - acc}")

print(f"Q5 smallest diff in accuracy with: number_of_courses_viewed")

# Q6
for C in [0.000001, 0.00001, 0.0001, 0.001]:
  acc = train_and_evaluate(df_train, y_train, df_val, y_val, None, C)
  print(f"C={C}: acc={acc} rounded={round(acc, 3)}")

print(f"Q6 Best accuracy for C=0.001")
