# =========================================================
# ML Zoomcamp - Homework 3 Solution
# =========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# Step 0: Grab the data
# ---------------------------------------------------------
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
df = pd.read_csv(url)

print(f"Data shape: {df.shape}")
df.head()

# ---------------------------------------------------------
# Step 1: Clean missing stuff as instructed
# ---------------------------------------------------------
# split features into cat vs num
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# take target out of feature lists
if 'converted' in num_cols:
    num_cols.remove('converted')
if 'converted' in cat_cols:
    cat_cols.remove('converted')

# fill missing: 'NA' for strings, 0.0 for numbers
df[cat_cols] = df[cat_cols].fillna('NA')
df[num_cols] = df[num_cols].fillna(0.0)

print("Missing values remaining:", df.isnull().sum().sum())

# ---------------------------------------------------------
# Q1: Mode of 'industry' column
# ---------------------------------------------------------
top_industry = df['industry'].mode()[0]
print(f"\n--- Q1 ---")
print(f"Most frequent industry is: {top_industry}")

# ---------------------------------------------------------
# Q2: Check correlation between numerical pairs
# ---------------------------------------------------------
corr = df[num_cols].corr()

pairs = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count')
]

print(f"\n--- Q2 ---")
max_c = -1
best_pair = None

for f1, f2 in pairs:
    val = abs(corr.loc[f1, f2])
    print(f"{f1} vs {f2}: {val:.4f}")
    if val > max_c:
        max_c = val
        best_pair = (f1, f2)

print(f">> Highest correlation: {best_pair[0]} & {best_pair[1]} ({max_c:.4f})")

# ---------------------------------------------------------
# Train / Val / Test Split (exact seed and ratios)
# ---------------------------------------------------------
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['converted'].values
y_val = df_val['converted'].values
y_test = df_test['converted'].values

# drop target from feature dataframes
del df_train['converted']
del df_val['converted']
del df_test['converted']

# ---------------------------------------------------------
# Q3: Mutual information for categorical features
# ---------------------------------------------------------
def get_mi(col):
    return mutual_info_score(df_train[col], y_train)

cats = ['industry', 'location', 'lead_source', 'employment_status']
mi_dict = {}

print(f"\n--- Q3 ---")
for c in cats:
    score = get_mi(c)
    mi_dict[c] = score
    print(f"{c}: {score:.4f} -> rounded: {round(score, 2)}")

best_mi = max(mi_dict, key=mi_dict.get)
print(f">> Variable with biggest MI score: {best_mi}")

# ---------------------------------------------------------
# Q4: Train baseline Logistic Regression with One-Hot Encoding
# ---------------------------------------------------------
all_feats = cat_cols + num_cols

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train[all_feats].to_dict(orient='records'))
X_val = dv.transform(df_val[all_feats].to_dict(orient='records'))

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
base_acc = accuracy_score(y_val, y_pred)

print(f"\n--- Q4 ---")
print(f"Validation Accuracy: {base_acc:.4f}")
print(f">> Rounded accuracy: {round(base_acc, 2)}")

# ---------------------------------------------------------
# Q5: Feature elimination (find least useful feature)
# ---------------------------------------------------------
feats_to_check = ['lead_source', 'number_of_courses_viewed', 'interaction_count']
diff_dict = {}

print(f"\n--- Q5 ---")
for feat in feats_to_check:
    sub_feats = [f for f in all_feats if f != feat]
    
    dv_sub = DictVectorizer(sparse=False)
    X_tr_sub = dv_sub.fit_transform(df_train[sub_feats].to_dict(orient='records'))
    X_val_sub = dv_sub.transform(df_val[sub_feats].to_dict(orient='records'))
    
    m_sub = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    m_sub.fit(X_tr_sub, y_train)
    
    acc_sub = accuracy_score(y_val, m_sub.predict(X_val_sub))
    diff = abs(base_acc - acc_sub)
    diff_dict[feat] = diff
    print(f"Without '{feat}': acc = {acc_sub:.4f} | diff = {diff:.4f}")

smallest_diff_feat = min(diff_dict, key=diff_dict.get)
print(f">> Feature with smallest difference: '{smallest_diff_feat}'")

# ---------------------------------------------------------
# Q6: Regularization tuning (C values)
# ---------------------------------------------------------
c_list = [0.000001, 0.00001, 0.0001, 0.001]
c_scores = {}

print(f"\n--- Q6 ---")
for C in c_list:
    m_c = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    m_c.fit(X_train, y_train)
    
    acc_c = accuracy_score(y_val, m_c.predict(X_val))
    acc_rounded = round(acc_c, 3)
    c_scores[C] = acc_rounded
    print(f"C={C:<8} | val_acc = {acc_c:.5f} | rounded (3 dec) = {acc_rounded:.3f}")

best_c = max(c_list, key=lambda c: (c_scores[c], -c))
print(f">> Best C value: {best_c}")