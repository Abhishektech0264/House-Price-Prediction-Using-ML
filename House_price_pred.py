import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn.datasets 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


df = pd.read_csv("C:\\Users\\bhosa\\MACHINE_LEARNING\\House-Price-Prediction-Using-ML\\Boston.csv", encoding='latin1')
print(df.head())

Housing_price_dataframe = df.copy()
print(Housing_price_dataframe.columns)

Housing_price_dataframe = Housing_price_dataframe.rename(columns={"medv":"price"} )
print(Housing_price_dataframe.columns)
print(Housing_price_dataframe.head())

Housing_price_dataframe.shape
Housing_price_dataframe.isnull().sum()

Housing_price_dataframe.describe()

correlation = Housing_price_dataframe.corr()

# Constructing heatmap to understan the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

X = Housing_price_dataframe.drop(['price'], axis=1)
Y = Housing_price_dataframe['price']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#Model traning

#XGBoost Regressor

model = XGBRegressor()
import xgboost
print(xgboost.__version__)

model = XGBRegressor()
model.fit(X_train, Y_train)

Traning_data_prediction = model.predict(X_train)
print(Traning_data_prediction)

score_1 = metrics.r2_score(Y_train, Traning_data_prediction)
print("R squared value = ", score_1)    
score_2 = metrics.mean_absolute_error(Y_train, Traning_data_prediction)
print("Mean Absolute Error = ", score_2)

plt.scatter(Y_train, Traning_data_prediction)
plt.xlabel("Actual Price")  
plt.ylabel("Predicted Price")  
plt.title("Actual Price vs Predicted Price")
plt.show()

Text_data_prediction = model.predict(X_test)
score_1 = metrics.r2_score(Y_test, Text_data_prediction)
print("R squared value = ", score_1)    
score_2 = metrics.mean_absolute_error(Y_test, Text_data_prediction)
print("Mean Absolute Error = ", score_2)