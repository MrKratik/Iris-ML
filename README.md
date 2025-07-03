# Iris ML Deployment App

This project demonstrates the deployment of a machine learning model for classifying Iris flower species using a Random Forest classifier. The app provides a simple interface for making predictions based on input features.

## Project Structure

- `app.py`: Main application file (likely a web server or API).
- `predictor.py`: Contains logic for loading the model and making predictions.
- `train_model.py`: Script to train the Random Forest model and save it as a pickle file.
- `model/iris_rf.pkl`: Pre-trained Random Forest model for Iris classification.
- `utils.py`: Utility functions used across the project.
- `sample_input.csv`: Example input data for testing predictions.
- `requirements.txt`: Python dependencies for the project.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MrKratik/Iris-ML.git
   cd Iris-ML
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python app.py
   ```

## Usage

- Use the provided `sample_input.csv` to test predictions.
- The app exposes endpoints (or a UI) for submitting Iris flower features and receiving predicted species.

## Model Details

- **Algorithm:** Random Forest Classifier
- **Dataset:** Iris dataset (sepal length, sepal width, petal length, petal width)
- **Output:** Predicted species (Setosa, Versicolor, Virginica)

## Author

- [Your Name]
- [Contact Information]

## License

This project is licensed under the MIT License.
