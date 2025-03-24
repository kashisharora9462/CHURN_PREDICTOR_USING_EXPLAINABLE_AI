import pandas as pd
data = {
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'SeniorCitizen': ['No', 'Yes', 'No', 'No', 'Yes'],
    'Partner': ['Yes', 'No', 'No', 'Yes', 'No'],
    'Dependents': ['Yes', 'No', 'No', 'Yes', 'No'],
    'tenure': [12, 1, 24, 36, 6],
    'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'MultipleLines': ['No', 'Yes', 'No phone service', 'Yes', 'No'],
    'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'],
    'OnlineSecurity': ['Yes', 'No', 'Yes', 'No internet service', 'No'],
    'OnlineBackup': ['No', 'Yes', 'Yes', 'No internet service', 'No'],
    'DeviceProtection': ['Yes', 'No', 'Yes', 'No internet service', 'Yes'],
    'TechSupport': ['No', 'No', 'Yes', 'No internet service', 'Yes'],
    'StreamingTV': ['Yes', 'Yes', 'No', 'No internet service', 'Yes'],
    'StreamingMovies': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
    'Contract': ['One year', 'Month-to-month', 'Two year', 'One year', 'Month-to-month'],
    'PaperlessBilling': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Electronic check', 'Mailed check', 'Credit card (automatic)', 'Electronic check'],
    'MonthlyCharges': [60.0, 100.0, 30.0, 20.0, 90.0],
    'TotalCharges': [720.0, 100.0, 720.0, 720.0, 540.0]
}
df = pd.DataFrame(data)
df.to_excel('demo_customers.xlsx', index=False)