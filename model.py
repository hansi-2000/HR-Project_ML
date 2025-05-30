# -*- coding: utf-8 -*-
"""Neural.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xwul5NzTswH8-b0hNr2d_WXzqI8bWkDd
"""

## IMPORT LIBRARIES ##

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Import the required function for preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# To save the model import pickle
import pickle

# train_url = 'https://drive.google.com/uc?export=download&id=1SlGTgYjSLKJsG41vFVvihZiGlRjBZff7'
df = pd.read_csv("Train.csv")


ordinal_mappings = {
    'Decision_skill_possess': {'Directive': 1, 'Behavioral': 2, 'Analytical': 3, 'Conceptual': 4},
    'Compensation_and_Benefits': {'type0': 1, 'type1': 2, 'type2': 3, 'type3': 4, 'type4':5}
}

for col, mapping in ordinal_mappings.items():
    df[col] = df[col].map(mapping)

df['Post_Level'].fillna(df['Post_Level'].median(), inplace=True)
df['Pay_Scale'].fillna(df['Pay_Scale'].median(), inplace=True)
df['Compensation_and_Benefits'].fillna(df['Compensation_and_Benefits'].mean(), inplace=True)

# Assuming your DataFrame is named df
df = df.dropna(subset=['Time_of_service'])
df = df.dropna(subset=['Age'])

df_to_modelling = df[['Compensation_and_Benefits','Decision_skill_possess','Education_Level', 'Time_of_service', 'Time_since_Salary_Increment', 'Distance_from_Home', 'Workload_Index', 'Pay_Scale', 'Post_Level', 'Growth_Rate', 'Yearly_Trainings', 'Weekly_Over_Time','Work_Life_Balance', 'Attrition_rate', 'Age']]
df_new =df_to_modelling

y = df_to_modelling['Attrition_rate']
df_to_modelling = df_to_modelling.drop(['Attrition_rate'], axis=1)
df_to_modelling['Pay_Scale'] = df_to_modelling['Pay_Scale'].astype(float)
df_to_modelling['Age'] = df_to_modelling['Age'].astype(int)
df_to_modelling['Post_Level'] = df_to_modelling['Post_Level'].astype(float)

# Define function to handle outliers
def clip_outliers(df, cols, threshold=1.5):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

# Apply to selected numerical features
num_features = ["Age", "Time_of_service", "Distance_from_Home", "Pay_Scale", "Yearly_Trainings"]  # Adjust based on data
df = clip_outliers(df, num_features)

scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

scaler = StandardScaler()
df_to_modelling_scaled = scaler.fit_transform(df_to_modelling)

# Cap values beyond 99th percentile
df['Distance_from_Home'] = np.clip(df['Distance_from_Home'], 0.05, 0.95)
df['VAR7'] = np.clip(df['Work_Life_Balance'], df['Work_Life_Balance'].quantile(0.05), df['Work_Life_Balance'].quantile(0.95))

X_train, X_test, y_train, y_test = train_test_split(df_to_modelling, y, test_size=0.2, random_state=42)

# Optimized Hyperparameters
gbm = lgb.LGBMRegressor(
    random_state=42,
    learning_rate=0.45,  # Reduce if needed
    max_depth=15,  # Reduce depth
    n_estimators=900,
    min_child_samples=25,  # Increase from 30
    colsample_bytree=0.8,
    subsample=0.8,
    lambda_l2=4 # Add back some L2 regularization
)

# Train the model
model = gbm.fit(df_to_modelling, y)

# Make predictions
# pred = model.predict(X_test)

pickle.dump(gbm, open("model.pkl", "wb"))