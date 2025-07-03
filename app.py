# app.py
import streamlit as st
import numpy as np
import pandas as pd
from predictor import load_model, predict
from utils import plot_confidence_bar, generate_prediction_report, plot_pairwise
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Deployment App 🚀", layout="wide")
st.title("🌸 Iris Classifier — Powered by Random Forest")

# Load model
model_path = "model/iris_rf.pkl"
model = load_model(model_path)
class_names = ['Setosa', 'Versicolor', 'Virginica']

# Choose input type
input_mode = st.radio("Choose input method:", ["Manual Input", "Upload CSV"], horizontal=True)

if input_mode == "Manual Input":
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
        petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
    with col2:
        sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
        petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

    input_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("🔍 Predict"):
        prediction, proba = predict(model, input_array)
        st.success(f"🎯 Prediction: {class_names[prediction]}")
        st.bar_chart(plot_confidence_bar(class_names, proba))

        report = generate_prediction_report(input_array, prediction, class_names, proba)
        st.download_button("📥 Download Report", report.to_csv(index=False), "report.csv", "text/csv")

else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())
        results = []
        probs = []

        for _, row in data.iterrows():
            pred, proba = predict(model, np.array([row]))
            results.append(class_names[pred])
            probs.append(max(proba))

        data["Prediction"] = results
        data["Confidence"] = probs
        st.success("✅ Batch predictions complete!")
        st.dataframe(data)

        st.download_button("📥 Download Batch Results", data.to_csv(index=False), "batch_results.csv", "text/csv")

# Visualization
st.subheader("📊 Visualize the Dataset")
iris = sns.load_dataset("iris")
viz_option = st.selectbox("Choose Plot", ["Pairplot", "Sepal vs Petal", "Species Count"])

if viz_option == "Pairplot":
    st.pyplot(plot_pairwise(iris))
elif viz_option == "Sepal vs Petal":
    fig, ax = plt.subplots()
    sns.scatterplot(data=iris, x="sepal_length", y="petal_length", hue="species", ax=ax)
    st.pyplot(fig)
else:
    fig, ax = plt.subplots()
    sns.countplot(data=iris, x="species", ax=ax)
    st.pyplot(fig)
