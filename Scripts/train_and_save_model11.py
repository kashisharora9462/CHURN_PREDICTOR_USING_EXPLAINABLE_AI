#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to pre-train an XGBoost model for telecom customer churn prediction and predict churn probability with SHAP analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import shap
import joblib
import warnings
import os

# Set configurations
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('display.max_columns', None)

# Path definitions
DATASET_PATH = "../Data/Raw/dataset.csv"
MODEL_PATH = "xgboost_model.joblib"
PREPROCESSOR_PATH = "preprocessor.joblib"
DATA_HEAD_PATH = "data_head.csv"
DATA_TYPES_PATH = "data_types.csv"
MISSING_VALUES_PATH = "missing_values.csv"
DUPLICATES_PATH = "duplicates.txt"
SUMMARY_STATS_PATH = "summary_stats.csv"
CHURN_DISTRIBUTION_PATH = "churn_distribution.csv"
PREPROCESSING_LOG_PATH = "preprocessing_log.txt"
CHURN_DISTRIBUTION_PLOT_PATH = "churn_distribution_plot.png"
CHURN_BY_GENDER_PLOT_PATH = "churn_by_gender_plot.png"
CHURN_BY_SENIOR_CITIZEN_PLOT_PATH = "churn_by_senior_citizen_plot.png"
CHURN_BY_CONTRACT_PLOT_PATH = "churn_by_contract_plot.png"
CORRELATION_MATRIX_PLOT_PATH = "correlation_matrix_plot.png"
FEATURE_ENGINEERING_LOG_PATH = "feature_engineering_log.txt"
TRAINING_LOG_PATH = "training_log.txt"
EVALUATION_METRICS_PATH = "evaluation_metrics.txt"
CONFUSION_MATRIX_PLOT_PATH = "confusion_matrix_plot.png"
ROC_CURVE_PLOT_PATH = "roc_curve_plot.png"
SHAP_SUMMARY_PATH = "shap_summary_plot.png"
SHAP_IMPORTANCE_PATH = "shap_importance_plot.png"
FEATURE_IMPORTANCE_PATH = "feature_importance.csv"
SHAP_FORCE_PLOT_PATH = "shap_force_plot.html"  # Changed to .html for interactive plot

# Existing functions (unchanged except where noted)
def load_and_explore_data(df):
    df.head().to_csv(DATA_HEAD_PATH, index=False)
    dtypes_df = pd.DataFrame(df.dtypes.astype(str), columns=['Data Type'])
    dtypes_df.to_csv(DATA_TYPES_PATH)
    missing_df = pd.DataFrame(df.isnull().sum().astype(str), columns=['Missing Values'])
    missing_df.to_csv(MISSING_VALUES_PATH)
    duplicates = df.duplicated().sum()
    with open(DUPLICATES_PATH, 'w') as f:
        f.write(str(duplicates))
    df.describe().to_csv(SUMMARY_STATS_PATH)
    churn_distribution = df['Churn'].value_counts(normalize=True) * 100
    churn_distribution_df = pd.DataFrame(churn_distribution.astype(str), columns=['Churn Distribution (%)'])
    churn_distribution_df.to_csv(CHURN_DISTRIBUTION_PATH)

def preprocess_data(df):
    df_processed = df.copy()
    preprocessing_log = []
    if 'TotalCharges' in df_processed.columns and df_processed['TotalCharges'].dtype == 'object':
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    missing_before = df_processed.isnull().sum().sum()
    preprocessing_log.append(f"Missing values before imputation: {missing_before}")
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        lower, upper = df_processed[col].quantile([0.01, 0.99]) if not df_processed[col].isna().all() else (0, 0)
        df_processed[col] = df_processed[col].clip(lower, upper)
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    missing_after = df_processed.isnull().sum().sum()
    preprocessing_log.append(f"Missing values after imputation: {missing_after}")
    if 'SeniorCitizen' in df_processed.columns:
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    if 'customerID' in df_processed.columns:
        df_processed.drop('customerID', axis=1, inplace=True)
    if 'Churn' in df_processed.columns and df_processed['Churn'].dtype == 'object':
        df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})
    preprocessing_log.append(f"Processed dataset has {df_processed.shape[0]} rows and {df_processed.shape[1]} columns.")
    with open(PREPROCESSING_LOG_PATH, 'w') as f:
        f.write("\n".join(preprocessing_log))
    object_cols = df_processed.select_dtypes(include=['object']).columns
    df_processed[object_cols] = df_processed[object_cols].astype(str)
    return df_processed

def perform_eda(df):
    if 'Churn' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Churn', data=df, ax=ax)
        plt.title('Churn Distribution')
        plt.savefig(CHURN_DISTRIBUTION_PLOT_PATH)
        plt.close()
    if 'gender' in df.columns and 'Churn' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='gender', hue='Churn', data=df, ax=ax)
        plt.title('Churn by Gender')
        plt.savefig(CHURN_BY_GENDER_PLOT_PATH)
        plt.close()
    if 'SeniorCitizen' in df.columns and 'Churn' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='SeniorCitizen', hue='Churn', data=df, ax=ax)
        plt.title('Churn by Senior Citizen Status')
        plt.savefig(CHURN_BY_SENIOR_CITIZEN_PLOT_PATH)
        plt.close()
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
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.title('Correlation Matrix')
    plt.savefig(CORRELATION_MATRIX_PLOT_PATH)
    plt.close()

def feature_engineering(df):
    df_fe = df.copy()
    df_fe['AvgMonthlyCost'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1)
    df_fe['HighRisk'] = ((df_fe['SeniorCitizen'] == 'Yes') & (df_fe['Contract'] == 'Month-to-month')).astype(int)
    df_fe['TenureBin'] = pd.cut(df_fe['tenure'], bins=[0, 12, 24, 36, 48, 60, 100], labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5+yr'])
    df_fe['ChargePerTenure'] = df_fe['MonthlyCharges'] * df_fe['tenure']
    with open(FEATURE_ENGINEERING_LOG_PATH, 'w') as f:
        f.write("Added features: AvgMonthlyCost, HighRisk, TenureBin, ChargePerTenure")
    return df_fe

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
    return X_train, X_test, y_train, y_test, preprocessor

def train_model(X_train, y_train, preprocessor):
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    X_train_sub_preprocessed = preprocessor.fit_transform(X_train_sub)
    X_val_preprocessed = preprocessor.transform(X_val)
    
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    
    xgb = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=10
    )
    
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 1000],
        'min_child_weight': [1, 2, 3, 4, 5, 6],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_lambda': [0, 0.1, 1, 10, 100],
        'reg_alpha': [0, 0.1, 1, 10]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=100,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(
        X_train_sub_preprocessed,
        y_train_sub,
        eval_set=[(X_val_preprocessed, y_val)],
        verbose=False
    )
    
    best_params = random_search.best_params_
    best_xgb = XGBClassifier(
        **best_params,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_xgb)])
    pipeline.fit(X_train, y_train)
    
    with open(TRAINING_LOG_PATH, 'w') as f:
        f.write(f"Best parameters: {random_search.best_params_}\n")
        f.write(f"Best CV Accuracy: {random_search.best_score_:.4f}\n")
    
    return pipeline

def evaluate_model(model, X_test, y_test):
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        accuracies = []
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            accuracies.append(accuracy_score(y_test, y_pred))
        best_threshold = thresholds[np.argmax(accuracies)]
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision_score_val = precision_score(y_test, y_pred)
        recall_score_val = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        metrics_log = [
            f"Best Threshold: {best_threshold:.4f}",
            f"Accuracy: {accuracy:.4f}",
            f"Precision: {precision_score_val:.4f}",
            f"Recall: {recall_score_val:.4f}",
            f"F1 Score: {f1:.4f}",
            f"ROC-AUC: {auc_score:.4f}"
        ]
        with open(EVALUATION_METRICS_PATH, 'w') as f:
            f.write("\n".join(metrics_log))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        plt.savefig(ROC_CURVE_PLOT_PATH)
        plt.close()
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        raise

def explain_model(model, X_test, preprocessor):
    preprocessed_X_test = preprocessor.transform(X_test)
    if hasattr(preprocessed_X_test, 'toarray'):
        preprocessed_X_test = preprocessed_X_test.toarray()
    
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values = explainer.shap_values(preprocessed_X_test)
    
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__')[-1] for name in feature_names]
    
    assert preprocessed_X_test.shape[1] == len(feature_names), "Feature count mismatch between data and names"
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, preprocessed_X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot for XGBoost')
    plt.savefig(SHAP_SUMMARY_PATH)
    plt.close()
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, preprocessed_X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance for XGBoost')
    plt.savefig(SHAP_IMPORTANCE_PATH)
    plt.close()
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.named_steps['model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

def precompute_steps():
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    load_and_explore_data(df)
    print("Step 1: Data exploration outputs saved.")
    
    df_processed = preprocess_data(df)
    print("Step 2: Preprocessing outputs saved.")
    
    perform_eda(df_processed)
    print("Step 3: EDA plots saved.")
    
    df_fe = feature_engineering(df_processed)
    print("Step 4: Feature engineering log saved.")
    
    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df_fe)
    print("Step 5: Features prepared.")
    
    model = train_model(X_train, y_train, preprocessor)
    print("Step 6: Training log saved.")
    
    evaluate_model(model, X_test, y_test)
    print("Step 7: Evaluation outputs saved.")
    
    explain_model(model, X_test, preprocessor)
    print("Step 8: SHAP outputs saved.")
    
    joblib.dump(model, MODEL_PATH,compress=3)
    joblib.dump(preprocessor, PREPROCESSOR_PATH,compress=3)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

# Enhanced prediction function with SHAP analysis
def predict_churn_with_shap(new_data, model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH, output_force_plot=SHAP_FORCE_PLOT_PATH):
    """
    Predict churn probability for new customer data and analyze feature contributions with SHAP.
    
    Args:
        new_data (dict or pd.DataFrame): Input data for prediction.
        model_path (str): Path to the saved model.
        preprocessor_path (str): Path to the saved preprocessor.
        output_force_plot (str): Path to save the SHAP force plot (HTML).
    
    Returns:
        tuple: (churn_probability, shap_values, feature_names)
    """
    try:
        # Load the trained model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Convert dictionary to DataFrame if necessary
        if isinstance(new_data, dict):
            new_df = pd.DataFrame([new_data])
        else:
            new_df = new_data.copy()
        
        # Apply feature engineering (same as training)
        new_df_fe = feature_engineering(new_df)
        
        # Ensure all expected columns are present (excluding 'Churn')
        expected_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                         'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 
                         'TotalCharges', 'AvgMonthlyCost', 'HighRisk', 'TenureBin', 'ChargePerTenure']
        for col in expected_cols:
            if col not in new_df_fe.columns:
                new_df_fe[col] = 0  # Default value, adjust as needed
        
        # Drop 'Churn' if present (not needed for prediction)
        if 'Churn' in new_df_fe.columns:
            new_df_fe = new_df_fe.drop('Churn', axis=1)
        
        # Predict probability
        churn_prob = model.predict_proba(new_df_fe)[:, 1][0]  # Probability of churn (class 1)
        
        # Preprocess the data for SHAP
        preprocessed_data = preprocessor.transform(new_df_fe)
        if hasattr(preprocessed_data, 'toarray'):
            preprocessed_data = preprocessed_data.toarray()
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model.named_steps['model'])
        shap_values = explainer.shap_values(preprocessed_data)
        
        # Get feature names
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.split('__')[-1] for name in feature_names]
        
        # Generate SHAP force plot (interactive HTML)
        shap.initjs()  # Initialize JavaScript for interactive plots
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            preprocessed_data[0], 
            feature_names=feature_names,
            show=False
        )
        shap.save_html(output_force_plot, force_plot)
        
        # Print results
        print(f"Churn Probability: {churn_prob:.4f} ({churn_prob*100:.2f}%)")
        print(f"SHAP force plot saved to: {output_force_plot}")
        
        return churn_prob, shap_values, feature_names
    
    except Exception as e:
        print(f"Error during prediction or SHAP analysis: {str(e)}")
        raise

# Example manual prediction with SHAP
def manual_prediction_with_shap_example():
    # Your manual input
    new_customer = {
        'gender': 'Male',
        'SeniorCitizen': 'No',
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 0,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 0.00,
        'TotalCharges': 0.00
    }
    
    # Predict churn probability and get SHAP analysis
    churn_prob, shap_values, feature_names = predict_churn_with_shap(new_customer)
    
    # Optional: Print top contributing features
    shap_contributions = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values[0]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)
    print("\nTop 5 Feature Contributions to Prediction:")
    print(shap_contributions.head())

if __name__ == "__main__":
    # Uncomment to train the model
    precompute_steps()
    
    # Run manual prediction with SHAP example
    manual_prediction_with_shap_example()