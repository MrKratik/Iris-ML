# utils.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confidence_bar(class_names, proba):
    df = pd.DataFrame({"Class": class_names, "Confidence": proba})
    return df.set_index("Class")

def generate_prediction_report(input_data, prediction, class_names, proba):
    df = pd.DataFrame(input_data, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
    df["Prediction"] = class_names[prediction]
    df["Confidence"] = max(proba)
    return df

def plot_pairwise(data):
    return sns.pairplot(data, hue="species")
