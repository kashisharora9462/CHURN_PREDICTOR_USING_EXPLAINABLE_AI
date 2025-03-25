# Telecom Customer Churn Prediction

## Overview
This project is a machine learning application designed to predict customer churn for a telecom company using a pre-trained Random Forest model. The application is built with Python and Streamlit, providing an interactive web interface for exploring model insights and predicting churn for new customers. Key features include:

- Data preprocessing and exploratory data analysis (EDA)
- Model training, evaluation, and interpretation using SHAP explainability
- A user-friendly interface for making predictions
- Business recommendations based on model insights

## Directory Structure
```
Telecom-Churn-Prediction/
|-- Data/
|   |-- Raw/
|   |   |-- dataset.csv  # Raw dataset
|   |-- Output/
|-- Models/
|   |-- preprocessor.joblib  # Preprocessing pipeline
|   |-- random_forest_model.joblib  # Trained model
|-- Plots/
|-- Logs/
|-- Scripts/
|   |-- train_and_save_model.py  # Model training script
|   |-- m1.py  # Streamlit application script
|   |-- main.py  # Alternative entry point for the app
|-- requirements.txt  # Dependencies
|-- README.md
```

## Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- Required Python packages listed in `requirements.txt`

## Setup Instructions
### 1. Clone the Repository
If this project is hosted in a repository, clone it to your local machine:
```sh
git clone <repository-url>
cd Telecom-Churn-Prediction
```

### 2. Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Prepare the Dataset
- Ensure the raw dataset (`dataset.csv`) is placed in the `Data/Raw/` directory.
- If missing, download a telecom churn dataset (e.g., from Kaggle) and place it in `Data/Raw/dataset.csv`.

### 5. Train the Model (If Model Files Are Missing)
If `Models/preprocessor.joblib` or `Models/random_forest_model.joblib` are missing, run:
```sh
python Scripts/train_and_save_model.py
```
This script also generates:
- Precomputed outputs in `Data/Output/`
- Logs in `Logs/`
- Plots in `Plots/`

## Running the Application
### 1. Launch the Streamlit App
Run the Streamlit application:
```sh
streamlit run Scripts/m1.py
```
Alternatively, you can run:
```sh
streamlit run Scripts/main.py
```

### 2. Access the App
- Open your browser and go to `http://localhost:8501` (or the URL provided by Streamlit).
- The app provides two main sections:
  - **Model Insights**: Explore data preprocessing, EDA, model evaluation, SHAP explainability, and business recommendations.
  - **Predict Churn**: Manually enter customer details or upload a batch file to predict churn probability.

## Usage
### Model Insights
- View machine learning pipeline steps (data loading, preprocessing, EDA, etc.).
- Download a PDF report summarizing insights.

### Predict Churn
- **Manual Input**: Enter customer details and see churn probability with SHAP feature contributions.
- **Batch Prediction**: Upload an Excel file with customer data to predict churn for multiple customers and download results as a CSV.
- **Feedback**: Use the sidebar to provide feedback on the app.

## Troubleshooting
- **Model or Preprocessor Not Found**:
  - Ensure `Models/preprocessor.joblib` and `Models/random_forest_model.joblib` exist. If missing, run `train_and_save_model.py`.
- **Dataset Not Found**:
  - Place `dataset.csv` in `Data/Raw/`.
- **Missing Precomputed Outputs**:
  - Run `train_and_save_model.py` to regenerate them.
- **Path Issues**:
  - Ensure scripts are executed from the correct directory, or update file paths accordingly.

## Dependencies
All required Python packages are listed in `requirements.txt`.

## Author
- Kashish Ahuja
- Contact: kashish.ahuja22b@iiitg.ac.in

## License
This project is licensed under the MIT License (or specify your preferred license).

