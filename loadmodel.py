
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle

df=pd.read_csv("../Harvestify-master/Data-processed/crop_recommendation.csv")

X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

XGBoost_model = XGBClassifier(random_state=2)

# Fit the model to the training data
XGBoost_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = XGBoost_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

with open('XGBoost_model.pkl', 'wb') as file:
    pickle.dump(XGBoost_model, file)