import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)

def predict_price(input_data):
    """
    input_data: list of 13 numeric features
    Returns: predicted house price
    """

    # Check number of features
    if len(input_data) != 13:
        raise ValueError(f"Expected 13 features, got {len(input_data)}")

    # Convert all inputs to float (handles strings like "6.5")
    try:
        input_array = np.array([float(x) for x in input_data]).reshape(1, -1)
    except ValueError:
        raise ValueError("All input features must be numeric (int or float)")

    # Predict
    prediction = model.predict(input_array)

    # Return single value
    return float(prediction[0])
