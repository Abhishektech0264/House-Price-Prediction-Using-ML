import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
df = pd.read_csv(os.path.join(BASE_DIR, "Boston.csv"))

# Rename target column
df = df.rename(columns={"medv": "price"})

# Split features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CREATE MODEL 👇 (THIS WAS MISSING)
model = XGBRegressor()
model.fit(X_train, y_train)

# Save model
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")
joblib.dump(model, MODEL_PATH)

print("✅ Model trained and saved successfully")
