## IMPORT LIBRARIES ##
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from httpx import request
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Import the required function for preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Import train and test split function
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Import Classifiers to be used
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
# Import packages to calculate performance of the models
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score

import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# To save the model import pickle
import pickle


# Load the CSV files
# Train dataset
train = pd.read_csv("train.csv")

# Test dataset
test = pd.read_csv("test.csv")

df = pd.concat([train, test])

# df = df.drop("Employee ID", axis=1)
df = df.drop("Gender", axis=1)
df = df.drop("Job Role", axis=1)
df = df.drop("Marital Status", axis=1) 

# train = train.drop("Employee ID", axis=1)
train = train.drop("Gender", axis=1)
train = train.drop("Job Role", axis=1)
train = train.drop("Marital Status", axis=1) 

# test = test.drop("Employee ID", axis=1)
test = test.drop("Gender", axis=1)
test = test.drop("Job Role", axis=1)
test = test.drop("Marital Status", axis=1)  


ordinal_mappings = {
    'Work-Life Balance': {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4},
    'Company Reputation': {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4},
    'Job Satisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
    'Performance Rating': {'Low': 1, 'Below Average': 2, 'Average': 3, 'High': 4},
    'Education Level': {'High School': 1, 'Associate Degree': 2, 'Bachelor’s Degree': 3, 'Master’s Degree': 4, 'PhD': 5},
    'Job Level': {'Entry': 1, 'Mid': 2, 'Senior': 3},
    'Company Reputation': {'Very Poor': 0, 'Poor': 1, 'Fair':2, 'Good': 3, 'Excellent': 4},
    'Employee Recognition': {'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3}
}


for col, mapping in ordinal_mappings.items():
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)

# print("Unique values in Gender column:", emp_data['Gender'].unique())
binary_columns = ['Innovation Opportunities', 'Overtime', 'Attrition', 'Remote Work', 'Leadership Opportunities']

binary_mapping = {'No': 0, 'Yes': 1, 'Stayed':1, 'Left':0}

for col in binary_columns:
    train[col] = train[col].map(binary_mapping)
    test[col] = test[col].map(binary_mapping)

df = pd.concat([train, test])
df.head(10)



# Drop unnecessary columns and define X and y
X = df.drop("Attrition", axis=1)
y = df["Attrition"]  # Ensure y is numeric

# model=Sequential()
# model.add(Dense(87,activation='relu'))
# model.add(Dense(76,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(53,activation='relu'))
# model.add(Dense(46,activation='relu'))
# model.add(Dense(35,activation='relu'))
# model.add(Dense(24,activation='relu'))
# model.add(Dense(12,activation='relu'))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(4,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=42)

# model1 = Sequential()
# model1.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
# model1.add(BatchNormalization())
# model1.add(Dropout(0.3))

# model1.add(Dense(32, activation='relu'))
# model1.add(BatchNormalization())
# model1.add(Dropout(0.3))

# model1.add(Dense(16, activation='relu'))
# model1.add(BatchNormalization())
# model1.add(Dropout(0.3))

# model1.add(Dense(1, activation='sigmoid'))

# model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model1.summary()


# history = model1.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=.20, verbose=1)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42
)

#classifier = model1

# Fit the model
classifier.fit(X_train, y_train)

predict = classifier.predict(X_test)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))