#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telecom Customer Churn Prediction with Streamlit (Automated Flow with Random Forest)
==================================================================================
This script creates an interactive Streamlit app for telecom customer churn prediction using a fixed dataset and a Random Forest model.
The app automates the entire process from data loading to recommendations.
"""

import streamlit as st
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
import warnings
import os

# Set configurations
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('display.max_columns', None)

# Path to the fixed dataset
DATASET_PATH = "dataset.csv"

# Function to load and explore data
def load_and_explore_data():
    st.write("### Step 1: Data Loading and Exploration")
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset file '{DATASET_PATH}' not found. Please ensure it is in the project directory.")
        return None
    df = pd.read_csv(DATASET_PATH)
    st.write(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Convert object columns to strings to avoid Arrow serialization issues
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].astype(str)
    
    st.write("#### First few rows of the dataset:")
    st.write(df.head())
    st.write("#### Data types:")
    # Convert dtypes to string to avoid Arrow serialization issues
    dtypes_str = df.dtypes.astype(str)
    st.write(dtypes_str)
    st.write("#### Missing values by column:")
    st.write(df.isnull().sum().astype(str))  # Convert to string
    duplicates = df.duplicated().sum()
    st.write(f"#### Number of duplicate rows: {duplicates}")
    st.write("#### Summary statistics for numerical columns:")
    st.write(df.describe())
    st.write("#### Class distribution (Churn):")
    churn_distribution = df['Churn'].value_counts(normalize=True) * 100
    st.write(churn_distribution.astype(str))  # Convert to string
    return df

# Function to preprocess data
def preprocess_data(df):
    st.write("### Step 2: Data Preprocessing")
    df_processed = df.copy()
    if 'TotalCharges' in df_processed.columns and df_processed['TotalCharges'].dtype == 'object':
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    missing_before = df_processed.isnull().sum().sum()
    st.write(f"Missing values before imputation: {missing_before}")
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    missing_after = df_processed.isnull().sum().sum()
    st.write(f"Missing values after imputation: {missing_after}")
    if 'SeniorCitizen' in df_processed.columns:
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    if 'customerID' in df_processed.columns:
        df_processed.drop('customerID', axis=1, inplace=True)
        st.write("Removed 'customerID' column.")
    if 'Churn' in df_processed.columns and df_processed['Churn'].dtype == 'object':
        df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})
        st.write("Converted 'Churn' to binary (0/1).")
    if 'tenure' in df_processed.columns:
        df_processed['tenure_group'] = pd.cut(df_processed['tenure'], 
                                              bins=[0, 12, 24, 36, 48, 60, 72], 
                                              labels=['0-12 months', '12-24 months', '24-36 months', 
                                                      '36-48 months', '48-60 months', '60+ months'])
        st.write("Created 'tenure_group' feature.")
    if 'TotalCharges' in df_processed.columns and 'tenure' in df_processed.columns:
        df_processed['AvgMonthlyCharges'] = df_processed['TotalCharges'] / (df_processed['tenure'] + 0.001)
        st.write("Created 'AvgMonthlyCharges' feature.")
    service_columns = [col for col in ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                                       'StreamingMovies'] if col in df_processed.columns]
    if service_columns:
        df_processed['ServiceCount'] = 0
        for col in service_columns:
            df_processed['ServiceCount'] += np.where(
                (df_processed[col] != 'No') & 
                (df_processed[col] != 'No internet service') & 
                (df_processed[col] != 'No phone service'), 
                1, 0)
        st.write("Created 'ServiceCount' feature.")
    st.write(f"Processed dataset has {df_processed.shape[0]} rows and {df_processed.shape[1]} columns.")
    
    # Convert object columns to strings to avoid Arrow serialization issues (precautionary)
    object_cols = df_processed.select_dtypes(include=['object']).columns
    df_processed[object_cols] = df_processed[object_cols].astype(str)
    
    return df_processed

# Function to perform EDA
def perform_eda(df):
    st.write("### Step 3: Exploratory Data Analysis")
    if 'Churn' in df.columns:
        st.write("#### Churn Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Churn', data=df, ax=ax)
        plt.title('Churn Distribution')
        st.pyplot(fig)
    
    if 'gender' in df.columns and 'Churn' in df.columns:
        st.write("#### Churn by Gender")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='gender', hue='Churn', data=df, ax=ax)
        plt.title('Churn by Gender')
        st.pyplot(fig)
    
    if 'SeniorCitizen' in df.columns and 'Churn' in df.columns:
        st.write("#### Churn by Senior Citizen")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='SeniorCitizen', hue='Churn', data=df, ax=ax)
        plt.title('Churn by Senior Citizen Status')
        st.pyplot(fig)
    
    if 'Contract' in df.columns and 'Churn' in df.columns:
        st.write("#### Churn Rate by Contract Type")
        fig, ax = plt.subplots(figsize=(8, 6))
        contract_churn = pd.crosstab(df['Contract'], df['Churn'])
        contract_churn_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
        contract_churn_pct['Churn Rate'] = contract_churn_pct[1]
        sns.barplot(x=contract_churn_pct.index, y='Churn Rate', data=contract_churn_pct, ax=ax)
        plt.title('Churn Rate by Contract Type')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.write("#### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.title('Correlation Matrix')
    st.pyplot(fig)

# Function to prepare features
def prepare_features(df):
    st.write("### Step 4: Feature Engineering and Preparation")
    if 'Churn' not in df.columns:
        st.error("Churn column not found in the dataset.")
        return None, None, None, None, None
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
    return X_train, X_test, y_train, y_test, preprocessor

# Function to train Random Forest model
def train_model(X_train, y_train, preprocessor):
    st.write("### Step 5: Model Training (Random Forest)")
    model = RandomForestClassifier(random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with st.spinner("Training Random Forest model..."):
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        pipeline.fit(X_train, y_train)
    st.write(f"Random Forest - Mean ROC-AUC: {cv_scores.mean():.4f}")
    return pipeline

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    st.write("### Step 6: Model Evaluation")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"ROC-AUC: {auc_score:.4f}")
    
    st.write("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    
    st.write("#### ROC Curve")
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)
    return accuracy, precision, recall, f1, auc_score

# Function to explain model with SHAP
def explain_model(model, X_train, X_test, preprocessor):
    st.write("### Step 7: Model Explainability with SHAP")
    preprocessed_X_test = preprocessor.transform(X_test)
    
    # Get feature names directly from the preprocessor
    feature_names = preprocessor.get_feature_names_out()
    # Clean up feature names by removing prefixes (e.g., 'num__', 'cat__')
    feature_names = [name.split('__')[-1] for name in feature_names]
    
    # Ensure preprocessed_X_test is a dense array (if sparse)
    if hasattr(preprocessed_X_test, 'toarray'):
        preprocessed_X_test = preprocessed_X_test.toarray()
    
    # Verify shapes
    st.write(f"Shape of preprocessed_X_test: {preprocessed_X_test.shape}")
    st.write(f"Length of feature_names: {len(feature_names)}")
    
    # Ensure preprocessed_X_test is a 2D array with shape (n_samples, n_features)
    if len(preprocessed_X_test.shape) != 2:
        st.error("preprocessed_X_test is not a 2D array.")
        return None
    
    n_samples, n_features = preprocessed_X_test.shape
    if n_samples != X_test.shape[0]:
        st.warning("preprocessed_X_test appears to be transposed. Transposing back to (n_samples, n_features).")
        preprocessed_X_test = preprocessed_X_test.T
        n_samples, n_features = preprocessed_X_test.shape
    
    model_to_explain = model.named_steps['model']
    with st.spinner("Calculating SHAP values..."):
        explainer = shap.TreeExplainer(model_to_explain)
        shap_values = explainer.shap_values(preprocessed_X_test)
    
    # Debug SHAP values
    st.write(f"Type of shap_values: {type(shap_values)}")
    if isinstance(shap_values, list):
        st.write(f"Number of elements in shap_values: {len(shap_values)}")
        for i, val in enumerate(shap_values):
            st.write(f"Shape of shap_values[{i}]: {val.shape}")
        # For binary classification, shap_values is a list: [class_0, class_1]
        if len(shap_values) == 2:
            shap_values = shap_values[1]  # Select class 1 (churn)
        else:
            st.error("Unexpected SHAP values format. Expected a list with 2 elements for binary classification.")
            return None
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        st.write(f"Shape of shap_values: {shap_values.shape}")
        # For binary classification, shap_values is a 3D array: (n_samples, n_features, n_classes)
        if shap_values.shape[2] == 2:  # Ensure there are 2 classes
            shap_values = shap_values[:, :, 1]  # Select class 1 (churn)
        else:
            st.error("Unexpected SHAP values shape. Expected 3D array with last dimension of size 2 for binary classification.")
            return None
    else:
        st.error("Unexpected SHAP values format. Expected either a list or a 3D NumPy array.")
        return None
    
    # Verify SHAP values shape after selection
    st.write(f"Shape of shap_values after selection: {shap_values.shape}")
    
    # Ensure shapes match
    if shap_values.shape[1] != preprocessed_X_test.shape[1]:
        st.error(f"Shape mismatch: shap_values has {shap_values.shape[1]} features, but preprocessed_X_test has {preprocessed_X_test.shape[1]} features.")
        return None
    
    if len(feature_names) != preprocessed_X_test.shape[1]:
        st.error(f"Feature names length ({len(feature_names)}) does not match number of features in preprocessed_X_test ({preprocessed_X_test.shape[1]}).")
        return None
    
    st.write("#### SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, preprocessed_X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot for Random Forest')
    st.pyplot(fig)
    
    st.write("#### SHAP Feature Importance")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, preprocessed_X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance for Random Forest')
    st.pyplot(fig)
    
    # Create feature importance DataFrame and ensure 'Feature' column is string
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model_to_explain.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance['Feature'] = feature_importance['Feature'].astype(str)  # Convert to string to avoid Arrow issues
    return feature_importance

# Function to generate business recommendations
def generate_recommendations(feature_importance, df_processed):
    if feature_importance is None:
        st.error("Cannot generate recommendations because feature importance could not be computed.")
        return
    st.write("### Step 8: Business Recommendations")
    st.write("#### Top features influencing customer churn:")
    st.write(feature_importance.head(10))
    st.write("#### Key Recommendations:")
    recommendations = [
        "1. Focus on contract types: Encourage longer-term contracts with incentives.",
        "2. Review pricing: Address high monthly charges linked to churn.",
        "3. Enhance service quality: Improve tech support and reliability.",
        "4. Target high-risk groups: Develop retention for senior citizens.",
        "5. Optimize payments: Review electronic payment methods.",
        "6. Investigate paperless billing: Understand its churn association.",
        "7. Bundle services: Offer attractive service packages.",
        "8. Early intervention: Identify at-risk customers early.",
        "9. Boost support: Enhance support for high-risk customers.",
        "10. Loyalty programs: Reward long-term customers."
    ]
    for rec in recommendations:
        st.write(rec)
    return recommendations

# Main Streamlit app
def main():
    st.title("Telecom Customer Churn Prediction with Explainable AI (Automated Flow)")
    st.write("This app automates the entire churn prediction process using a Random Forest model on the fixed dataset `dataset.csv`.")
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Step 2: Preprocess data
    df_processed = preprocess_data(df)
    
    # Step 3: Perform EDA
    perform_eda(df_processed)
    
    # Step 4: Prepare features
    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df_processed)
    if X_train is None:
        return
    
    # Step 5: Train Random Forest model
    model = train_model(X_train, y_train, preprocessor)
    
    # Step 6: Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Step 7: Explain model with SHAP
    feature_importance = explain_model(model, X_train, X_test, preprocessor)
    
    # Step 8: Generate business recommendations
    generate_recommendations(feature_importance, df_processed)

if __name__ == "__main__":
    main()