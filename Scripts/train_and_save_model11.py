#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to pre-train the Random Forest model for telecom customer churn prediction with 22 raw features,
precompute outputs for Steps 1-8, and save them for display in the Streamlit app.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
import warnings
import os

# Set configurations
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('display.max_columns', None)

# Path to the dataset (must contain all 22 features + Churn + customerID)
DATASET_PATH = "dataset.csv"

# Paths to save the model, preprocessor, and precomputed outputs
MODEL_PATH = "random_forest_model.joblib"
PREPROCESSOR_PATH = "preprocessor.joblib"

# Step 1: Data Loading and Exploration
DATA_HEAD_PATH = "data_head.csv"
DATA_TYPES_PATH = "data_types.csv"
MISSING_VALUES_PATH = "missing_values.csv"
DUPLICATES_PATH = "duplicates.txt"
SUMMARY_STATS_PATH = "summary_stats.csv"
CHURN_DISTRIBUTION_PATH = "churn_distribution.csv"

# Step 2: Data Preprocessing
PREPROCESSING_LOG_PATH = "preprocessing_log.txt"

# Step 3: EDA
CHURN_DISTRIBUTION_PLOT_PATH = "churn_distribution_plot.png"
CHURN_BY_GENDER_PLOT_PATH = "churn_by_gender_plot.png"
CHURN_BY_SENIOR_CITIZEN_PLOT_PATH = "churn_by_senior_citizen_plot.png"
CHURN_BY_CONTRACT_PLOT_PATH = "churn_by_contract_plot.png"
CORRELATION_MATRIX_PLOT_PATH = "correlation_matrix_plot.png"

# Step 4: Feature Engineering
FEATURE_ENGINEERING_LOG_PATH = "feature_engineering_log.txt"

# Step 5: Model Training
TRAINING_LOG_PATH = "training_log.txt"

# Step 6: Model Evaluation
EVALUATION_METRICS_PATH = "evaluation_metrics.txt"
CONFUSION_MATRIX_PLOT_PATH = "confusion_matrix_plot.png"
ROC_CURVE_PLOT_PATH = "roc_curve_plot.png"

# Step 7: SHAP Explainability
SHAP_SUMMARY_PATH = "shap_summary_plot.png"
SHAP_IMPORTANCE_PATH = "shap_importance_plot.png"
FEATURE_IMPORTANCE_PATH = "feature_importance.csv"

# Function to load and explore data (Step 1)
def load_and_explore_data(df):
    # Save first few rows
    df.head().to_csv(DATA_HEAD_PATH, index=False)
    
    # Save data types
    dtypes_str = df.dtypes.astype(str)
    dtypes_df = pd.DataFrame(dtypes_str, columns=['Data Type'])
    dtypes_df.to_csv(DATA_TYPES_PATH)
    
    # Save missing values
    missing_values = df.isnull().sum().astype(str)
    missing_df = pd.DataFrame(missing_values, columns=['Missing Values'])
    missing_df.to_csv(MISSING_VALUES_PATH)
    
    # Save duplicates
    duplicates = df.duplicated().sum()
    with open(DUPLICATES_PATH, 'w') as f:
        f.write(str(duplicates))
    
    # Save summary statistics
    df.describe().to_csv(SUMMARY_STATS_PATH)
    
    # Save churn distribution
    churn_distribution = df['Churn'].value_counts(normalize=True) * 100
    churn_distribution_df = pd.DataFrame(churn_distribution.astype(str), columns=['Churn Distribution (%)'])
    churn_distribution_df.to_csv(CHURN_DISTRIBUTION_PATH)

# Function to preprocess data (Step 2)
def preprocess_data(df):
    df_processed = df.copy()
    preprocessing_log = []
    
    # Convert TotalCharges to numeric if needed
    if 'TotalCharges' in df_processed.columns and df_processed['TotalCharges'].dtype == 'object':
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    missing_before = df_processed.isnull().sum().sum()
    preprocessing_log.append(f"Missing values before imputation: {missing_before}")
    
    # Impute missing values
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    missing_after = df_processed.isnull().sum().sum()
    preprocessing_log.append(f"Missing values after imputation: {missing_after}")
    
    # Convert SeniorCitizen to categorical
    if 'SeniorCitizen' in df_processed.columns:
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        preprocessing_log.append("Converted 'SeniorCitizen' to categorical.")
    
    # Remove customerID if present
    if 'customerID' in df_processed.columns:
        df_processed.drop('customerID', axis=1, inplace=True)
        preprocessing_log.append("Removed 'customerID' column.")
    
    # Convert Churn to binary
    if 'Churn' in df_processed.columns and df_processed['Churn'].dtype == 'object':
        df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})
        preprocessing_log.append("Converted 'Churn' to binary (0/1).")
    
    preprocessing_log.append(f"Processed dataset has {df_processed.shape[0]} rows and {df_processed.shape[1]} columns.")
    
    # Save preprocessing log
    with open(PREPROCESSING_LOG_PATH, 'w') as f:
        f.write("\n".join(preprocessing_log))
    
    # Convert object columns to strings
    object_cols = df_processed.select_dtypes(include=['object']).columns
    df_processed[object_cols] = df_processed[object_cols].astype(str)
    
    return df_processed

# Function to perform EDA (Step 3)
def perform_eda(df):
    # Churn Distribution
    if 'Churn' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Churn', data=df, ax=ax)
        plt.title('Churn Distribution')
        plt.savefig(CHURN_DISTRIBUTION_PLOT_PATH)
        plt.close()
    
    # Churn by Gender
    if 'gender' in df.columns and 'Churn' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='gender', hue='Churn', data=df, ax=ax)
        plt.title('Churn by Gender')
        plt.savefig(CHURN_BY_GENDER_PLOT_PATH)
        plt.close()
    
    # Churn by Senior Citizen
    if 'SeniorCitizen' in df.columns and 'Churn' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='SeniorCitizen', hue='Churn', data=df, ax=ax)
        plt.title('Churn by Senior Citizen Status')
        plt.savefig(CHURN_BY_SENIOR_CITIZEN_PLOT_PATH)
        plt.close()
    
    # Churn Rate by Contract Type
    if 'Contract' in df.columns and 'Churn' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        contract_churn = pd.crosstab(df['Contract'], df['Churn'])
        contract_churn_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
        contract_churn_pct['Churn Rate'] = contract_churn_pct[1]
        sns.barplot(x=contract_churn_pct.index, y='Churn Rate', data=contract_churn_pct, ax=ax)
        plt.title('Churn Rate by Contract Type')
        plt.xticks(rotation=45)
        plt.savefig(CHURN_BY_CONTRACT_PLOT_PATH)
        plt.close()
    
    # Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.title('Correlation Matrix')
    plt.savefig(CORRELATION_MATRIX_PLOT_PATH)
    plt.close()

# Function to prepare features (Step 4)
def prepare_features(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Save feature engineering log
    with open(FEATURE_ENGINEERING_LOG_PATH, 'w') as f:
        f.write("Features prepared successfully with 22 raw inputs!")
    
    return X_train, X_test, y_train, y_test, preprocessor

# Function to train model (Step 5)
def train_model(X_train, y_train, preprocessor):
    model = RandomForestClassifier(random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    pipeline.fit(X_train, y_train)
    
    # Save training log
    with open(TRAINING_LOG_PATH, 'w') as f:
        f.write(f"Random Forest - Mean ROC-AUC: {cv_scores.mean():.4f}")
    
    return pipeline

# Function to evaluate model (Step 6)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Save evaluation metrics
    metrics_log = [
        f"Accuracy: {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1 Score: {f1:.4f}",
        f"ROC-AUC: {auc_score:.4f}"
    ]
    with open(EVALUATION_METRICS_PATH, 'w') as f:
        f.write("\n".join(metrics_log))
    
    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
    plt.close()
    
    # Save ROC curve plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    plt.savefig(ROC_CURVE_PLOT_PATH)
    plt.close()

# Function to explain model with SHAP (Step 7)
def explain_model(model, X_test, preprocessor):
    preprocessed_X_test = preprocessor.transform(X_test)
    if hasattr(preprocessed_X_test, 'toarray'):
        preprocessed_X_test = preprocessed_X_test.toarray()
    
    model_to_explain = model.named_steps['model']
    explainer = shap.TreeExplainer(model_to_explain)
    shap_values = explainer.shap_values(preprocessed_X_test)
    
    # Handle SHAP values format
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]  # Select class 1 (churn)
        else:
            print(f"Unexpected SHAP values format: {len(shap_values)} elements. Expected 2 for binary classification.")
            return
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__')[-1] for name in feature_names]
    
    # Debugging: Check shapes and types
    print(f"Shape of preprocessed_X_test: {preprocessed_X_test.shape}")
    print(f"Shape of shap_values: {shap_values.shape}")
    print(f"Length of feature_names: {len(feature_names)}")
    print(f"Type of feature_names: {type(feature_names)}")
    print(f"First few feature_names: {feature_names[:5]}")
    
    # Ensure feature_names length matches the number of features
    expected_num_features = preprocessed_X_test.shape[1]
    if len(feature_names) != expected_num_features:
        print(f"Error: Length of feature_names ({len(feature_names)}) does not match number of features in preprocessed_X_test ({expected_num_features}).")
        return
    
    # Convert feature_names to a NumPy array to ensure proper indexing
    feature_names = np.array(feature_names)
    
    # Save SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, preprocessed_X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot for Random Forest')
    plt.savefig(SHAP_SUMMARY_PATH)
    plt.close()
    
    # Save SHAP feature importance plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, preprocessed_X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance for Random Forest')
    plt.savefig(SHAP_IMPORTANCE_PATH)
    plt.close()
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model_to_explain.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

# Main function to precompute all steps
def precompute_steps():
    # Load data
    df = pd.read_csv('dataset.csv')
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Step 1: Data Loading and Exploration
    load_and_explore_data(df)
    print("Step 1: Data exploration outputs saved.")
    
    # Step 2: Preprocess data
    df_processed = preprocess_data(df)
    print("Step 2: Preprocessing outputs saved.")
    
    # Step 3: Perform EDA
    perform_eda(df_processed)
    print("Step 3: EDA plots saved.")
    
    # Step 4: Prepare features
    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df_processed)
    print("Step 4: Feature engineering log saved.")
    
    # Step 5: Train model
    model = train_model(X_train, y_train, preprocessor)
    print("Step 5: Training log saved.")
    
    # Step 6: Evaluate model
    evaluate_model(model, X_test, y_test)
    print("Step 6: Evaluation outputs saved.")
    
    # Step 7: Explain model with SHAP
    explain_model(model, X_test, preprocessor)
    print("Step 7: SHAP outputs saved.")
    
    # Save the model and preprocessor
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    precompute_steps()