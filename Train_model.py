import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

df = pd.read_csv("Boston.csv")

df = df.rename(columns={"medv": "price"})

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

model = XGBRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "house_price_model.pkl")

print("Model trained and saved successfully")
