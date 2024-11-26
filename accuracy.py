import pandas as pd
import pickle
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data1= pd.read_csv('ad_click_records.csv')
data2= pd.read_csv('adclick.csv')
X_train = data2[['Daily Time', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y_train = data2['Clicked on Ad']
X_test = data2[['Daily Time', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y_test = data2['Clicked on Ad']
model=LogisticRegression()

scaler = StandardScaler()
X_test=scaler.fit_transform(X_test)
X_train=scaler.fit_transform(X_test)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print('accuracy_score = ',accuracy_score(y_test,y_pred))
print("\nClassification Report (Test Set):\n", classification_report(y_test, y_pred))
