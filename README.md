# 🌸 Iris ML Deployment App

A beautifully interactive **machine learning web app** built with [Streamlit](https://streamlit.io/) that predicts **Iris flower species** based on user inputs or batch CSV upload. This project showcases the **end-to-end deployment of a trained Random Forest Classifier** with real-time predictions, model confidence visualization, and powerful data insights.

---

## 🔗 Live Demo  
**[🌐 View on Streamlit Cloud →](https://celebal-ml-devlopment.streamlit.app/)**  
_(No installation needed — try it instantly!)_

---

## 🚀 Features

- ✅ Manual & Batch prediction modes  
- ✅ Real-time output with prediction confidence  
- ✅ Interactive visualizations (scatter plots, pair plots)  
- ✅ Downloadable prediction reports  
- ✅ Modular code for scalability & clean architecture  
- ✅ Powered by Streamlit + scikit-learn

---

## 📁 Project Structure

Iris-ML/
│
├── app.py                  # 🌐 Main Streamlit web app
├── train_model.py          # 🧠 Script to train & save Random Forest model
├── predictor.py            # 🔮 Loads model and returns predictions
├── utils.py                # 📊 Charts & report helpers
├── sample_input.csv        # 📥 Sample input CSV for batch prediction
├── requirements.txt        # 📦 Python dependencies
├── README.md               # 📘 Project documentation
└── model/
    └── iris_rf.pkl         # ✅ Trained Random Forest model


yaml
Copy
Edit

---

## ⚙️ Tech Stack

| Layer        | Tech                  |
|--------------|------------------------|
| Frontend     | Streamlit              |
| ML Framework | scikit-learn           |
| Model Type   | Random Forest Classifier |
| Visualization| Seaborn, Matplotlib    |
| Format       | `.pkl` for ML model    |

---

## 🧠 Model Info

| Detail         | Value                       |
|----------------|-----------------------------|
| **Dataset**    | Iris Dataset (150 samples)  |
| **Features**   | Sepal & Petal Length/Width  |
| **Algorithm**  | Random Forest Classifier    |
| **Accuracy**   | ~97% on test set            |
| **Output**     | Iris species (Setosa, Versicolor, Virginica) |

---

## 📸 Screenshots *(optional)*  
>![image](https://github.com/user-attachments/assets/fb06dce0-2089-4609-8286-53747b67bd37)
![image](https://github.com/user-attachments/assets/e47b9cbd-8e0c-40e5-ba13-b5d48eaee85d)
![image](https://github.com/user-attachments/assets/12de46f8-5a03-4148-8845-5b6cf92a8c81)




---

## 🛠️ Local Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MrKratik/Iris-ML.git
   cd Iris-ML
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Train the Model (optional, already included)

bash
Copy
Edit
python train_model.py
Run the App

bash
Copy
Edit
streamlit run app.py
🧪 Usage
Use the web interface to enter values manually OR upload a CSV file.

Test with the provided sample_input.csv file.

View real-time predictions with confidence scores.

Download results for offline use.

🌐 Deploy on Streamlit Cloud (Free)
Push code to a public GitHub repo (✅ already done).

Go to https://streamlit.io/cloud

Click "New App" and connect your repo: MrKratik/Iris-ML

Set main file to app.py, and branch to main

Click "Deploy"

Visit your live app:
👉 https://celebal-ml-devlopment.streamlit.app/

👨‍💻 Author
Kratik Jain
📧 kratikjain121@gmail.com
🔗 https://github.com/MrKratik/Iris-ML

📄 License
This project is licensed under the MIT License.
