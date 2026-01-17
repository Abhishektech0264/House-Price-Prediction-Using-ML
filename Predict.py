import joblib
import numpy as np

model = joblib.load("house_price_model.pkl")

def predict_price(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]
