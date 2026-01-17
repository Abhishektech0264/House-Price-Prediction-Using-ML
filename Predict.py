import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

joblib.dump(model, MODEL_PATH)
