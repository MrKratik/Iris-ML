# predictor.py
import joblib
import numpy as np

def load_model(path):
    return joblib.load(path)

def predict(model, input_array):
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]
    return prediction, proba
