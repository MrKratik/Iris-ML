# train_model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/iris_rf.pkl")
print("✅ Model saved at model/iris_rf.pkl")
