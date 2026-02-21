import joblib #joblib is used to save and load trained machine learning models.
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   

REGRESSOR_PATH = os.path.join(BASE_DIR, "models", "regressor.pkl")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "classifier.pkl")

print("Loading ML models...")

regressor = joblib.load(REGRESSOR_PATH)
classifier = joblib.load(CLASSIFIER_PATH)

print("Models loaded successfully.")