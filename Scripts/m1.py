#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telecom Customer Churn Prediction with Streamlit (Pre-trained XGBoost)
==================================================================================
This script creates an enhanced Streamlit app for telecom customer churn prediction using a pre-trained XGBoost model.
The app features a Home Page with navigation options, improved visual design, interactive elements, and additional functionality including batch prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import joblib
import warnings
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Set configurations
warnings.filterwarnings('ignore')
np.random.seed(42)


MODEL_PATH = "../Models/xgboost_model.joblib"
PREPROCESSOR_PATH = "../Models/preprocessor.joblib"
DATASET_PATH = "../Data/Raw/dataset.csv"




# Step 1: Data Loading and Exploration
DATA_HEAD_PATH = "../Data/Output/data_head.csv"
DATA_TYPES_PATH = "../Data/Output/data_types.csv"
MISSING_VALUES_PATH = "../Data/Output/missing_values.csv"
DUPLICATES_PATH = "../Data/Output/duplicates.txt"
SUMMARY_STATS_PATH = "../Data/Output/summary_stats.csv"
CHURN_DISTRIBUTION_PATH = "../Data/Output/churn_distribution.csv"

# Step 2: Data Preprocessing
PREPROCESSING_LOG_PATH = "../Logs/preprocessing_log.txt"

# Step 3: EDA
CHURN_DISTRIBUTION_PLOT_PATH = "../Plots/churn_distribution_plot.png"
CHURN_BY_GENDER_PLOT_PATH = "../Plots/churn_by_gender_plot.png"
CHURN_BY_SENIOR_CITIZEN_PLOT_PATH = "../Plots/churn_by_senior_citizen_plot.png"
CHURN_BY_CONTRACT_PLOT_PATH = "../Plots/churn_by_contract_plot.png"
CORRELATION_MATRIX_PLOT_PATH = "../Plots/correlation_matrix_plot.png"

# Step 4: Feature Engineering
FEATURE_ENGINEERING_LOG_PATH = "../Logs/feature_engineering_log.txt"

# Step 5: Model Training
TRAINING_LOG_PATH = "../Logs/training_log.txt"

# Step 6: Model Evaluation
EVALUATION_METRICS_PATH = "../Logs/evaluation_metrics.txt"
CONFUSION_MATRIX_PLOT_PATH = "../Plots/confusion_matrix_plot.png"
ROC_CURVE_PLOT_PATH = "../Plots/roc_curve_plot.png"

# Step 7: SHAP Explainability
SHAP_SUMMARY_PATH = "../Plots/shap_summary_plot.png"
SHAP_IMPORTANCE_PATH = "../Plots/shap_importance_plot.png"
FEATURE_IMPORTANCE_PATH = "../Data/Output/feature_importance.csv"



@st.cache_resource
def load_model_and_preprocessor():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        st.error(f"❌ Model or preprocessor file not found at {MODEL_PATH} and {PREPROCESSOR_PATH}. Please run 'train_and_save_model11.py' to train and save the model.")
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        classifier = model.named_steps['model']
        return classifier, preprocessor
    except Exception as e:
        st.error(f"❌ Error loading model or preprocessor: {str(e)}")
        return None, None

@st.cache_data
def load_precomputed_data(file_path, file_type="csv"):
    try:
        if file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "txt":
            with open(file_path, 'r') as f:
                return f.read()
        elif file_type == "lines":
            with open(file_path, 'r') as f:
                return f.read().splitlines()
    except Exception as e:
        st.warning(f"⚠️ Error loading {file_path}: {str(e)}")
        return None

def generate_pdf_report(content_dict, title="Report"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y_position = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, title)
    y_position -= 30
    c.setFont("Helvetica", 12)
    
    for section, content in content_dict.items():
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, section)
        y_position -= 20
        c.setFont("Helvetica", 12)
        if isinstance(content, pd.DataFrame):
            content_str = content.to_string(index=False)
        else:
            content_str = str(content)
        lines = content_str.split('\n')
        for line in lines:
            if y_position < 50:
                c.showPage()
                y_position = height - 50
            c.drawString(50, y_position, line[:100])
            y_position -= 15
    
    c.save()
    buffer.seek(0)
    return buffer

def display_step_1():
    with st.expander("📊 Step 1: Data Loading and Exploration", expanded=False):
        try:
            df = pd.read_csv(DATASET_PATH)
            st.success(f"✅ Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"❌ Error loading dataset from {DATASET_PATH}: {str(e)}")
            return
        
        data_head = load_precomputed_data(DATA_HEAD_PATH)
        if data_head is not None:
            st.markdown("#### 🔍 First few rows of the dataset:")
            st.write(data_head)
        
        data_types = load_precomputed_data(DATA_TYPES_PATH)
        if data_types is not None:
            st.markdown("#### 📋 Data types:")
            st.write(data_types)
        
        missing_values = load_precomputed_data(MISSING_VALUES_PATH)
        if missing_values is not None:
            st.markdown("#### 🕳️ Missing values by column:")
            st.write(missing_values)
        
        duplicates = load_precomputed_data(DUPLICATES_PATH, file_type="txt")
        if duplicates is not None:
            st.markdown(f"#### 🔄 Number of duplicate rows: {duplicates}")
        
        summary_stats = load_precomputed_data(SUMMARY_STATS_PATH)
        if summary_stats is not None:
            st.markdown("#### 📈 Summary statistics for numerical columns:")
            st.write(summary_stats)
        
        churn_distribution = load_precomputed_data(CHURN_DISTRIBUTION_PATH)
        if churn_distribution is not None:
            st.markdown("#### ⚖️ Class distribution (Churn):")
            st.write(churn_distribution)

def display_step_2():
    with st.expander("🛠️ Step 2: Data Preprocessing", expanded=False):
        preprocessing_log = load_precomputed_data(PREPROCESSING_LOG_PATH, file_type="lines")
        if preprocessing_log is not None:
            for log in preprocessing_log:
                st.write(log)

def display_step_3():
    with st.expander("📈 Step 3: Exploratory Data Analysis", expanded=False):
        if os.path.exists(CHURN_DISTRIBUTION_PLOT_PATH):
            st.markdown("#### 📊 Churn Distribution")
            st.image(CHURN_DISTRIBUTION_PLOT_PATH, caption="Churn Distribution", use_container_width=True)
        
        if os.path.exists(CHURN_BY_GENDER_PLOT_PATH):
            st.markdown("#### 👥 Churn by Gender")
            st.image(CHURN_BY_GENDER_PLOT_PATH, caption="Churn by Gender", use_container_width=True)
        
        if os.path.exists(CHURN_BY_SENIOR_CITIZEN_PLOT_PATH):
            st.markdown("#### 🧓 Churn by Senior Citizen")
            st.image(CHURN_BY_SENIOR_CITIZEN_PLOT_PATH, caption="Churn by Senior Citizen Status", use_container_width=True)
        
        if os.path.exists(CHURN_BY_CONTRACT_PLOT_PATH):
            st.markdown("#### 📜 Churn Rate by Contract Type")
            st.image(CHURN_BY_CONTRACT_PLOT_PATH, caption="Churn Rate by Contract Type", use_container_width=True)
        
        if os.path.exists(CORRELATION_MATRIX_PLOT_PATH):
            st.markdown("#### 🌡️ Correlation Matrix")
            st.image(CORRELATION_MATRIX_PLOT_PATH, caption="Correlation Matrix", use_container_width=True)

def display_step_4():
    with st.expander("🛠️ Step 4: Feature Engineering and Preparation", expanded=False):
        feature_engineering_log = load_precomputed_data(FEATURE_ENGINEERING_LOG_PATH, file_type="txt")
        if feature_engineering_log is not None:
            st.success(feature_engineering_log)

def display_step_5():
    with st.expander("🏋️ Step 5: Model Training (XGBoost)", expanded=False):
        training_log = load_precomputed_data(TRAINING_LOG_PATH, file_type="txt")
        if training_log is not None:
            st.success(f"✅ {training_log}")

def display_step_6():
    with st.expander("📊 Step 6: Model Evaluation", expanded=False):
        metrics_log = load_precomputed_data(EVALUATION_METRICS_PATH, file_type="lines")
        if metrics_log is not None:
            for metric in metrics_log:
                st.write(metric)
        
        if os.path.exists(CONFUSION_MATRIX_PLOT_PATH):
            st.markdown("#### 🖼️ Confusion Matrix")
            st.image(CONFUSION_MATRIX_PLOT_PATH, caption="Confusion Matrix", use_container_width=True)
        
        if os.path.exists(ROC_CURVE_PLOT_PATH):
            st.markdown("#### 📈 ROC Curve")
            st.image(ROC_CURVE_PLOT_PATH, caption="ROC Curve", use_container_width=True)

def display_step_7():
    with st.expander("🔍 Step 7: Model Explainability with SHAP", expanded=False):
        if os.path.exists(SHAP_SUMMARY_PATH):
            st.markdown("#### 📊 SHAP Summary Plot")
            st.image(SHAP_SUMMARY_PATH, caption="SHAP Summary Plot for XGBoost", use_container_width=True)
        
        if os.path.exists(SHAP_IMPORTANCE_PATH):
            st.markdown("#### 📈 SHAP Feature Importance")
            st.image(SHAP_IMPORTANCE_PATH, caption="SHAP Feature Importance for XGBoost", use_container_width=True)
        
        feature_importance = load_precomputed_data(FEATURE_IMPORTANCE_PATH)
        if feature_importance is not None:
            st.markdown("#### 📋 Feature Importance")
            st.write(feature_importance.head(10))

def display_step_8():
    with st.expander("💡 Step 8: Business Recommendations", expanded=False):
        st.markdown("#### 📋 Top features influencing customer churn:")
        feature_importance = load_precomputed_data(FEATURE_IMPORTANCE_PATH)
        if feature_importance is not None:
            st.write(feature_importance.head(10))
        
        st.markdown("#### 🗣️ Key Recommendations:")
        recommendations = [
            "1. 📜 **Focus on contract types**: Encourage longer-term contracts with incentives.",
            "2. 💰 **Review pricing**: Address high monthly charges linked to churn.",
            "3. 🛠️ **Enhance service quality**: Improve tech support and reliability.",
            "4. 🧓 **Target high-risk groups**: Develop retention for senior citizens.",
            "5. 💳 **Optimize payments**: Review electronic payment methods.",
            "6. 📄 **Investigate paperless billing**: Understand its churn association.",
            "7. 📦 **Bundle services**: Offer attractive service packages.",
            "8. ⏰ **Early intervention**: Identify at-risk customers early.",
            "9. 📞 **Boost support**: Enhance support for high-risk customers.",
            "10. 🎁 **Loyalty programs**: Reward long-term customers."
        ]
        for rec in recommendations:
            st.write(rec)

def preprocess_data(df, required_features):
    df_processed = df.copy()
    if 'TotalCharges' in df_processed.columns and df_processed['TotalCharges'].dtype == 'object':
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    numeric_cols = df_processed.select_dtypes(include=['number']).columns.intersection(required_features)
    for col in numeric_cols:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.intersection(required_features)
    for col in categorical_cols:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    if 'SeniorCitizen' in df_processed.columns:
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({0: 'No', 1: 'Yes', 'No': 'No', 'Yes': 'Yes'})
    if 'customerID' in df_processed.columns:
        df_processed.drop('customerID', axis=1, inplace=True)
    # Add engineered features
    df_processed['AvgMonthlyCost'] = df_processed['TotalCharges'] / (df_processed['tenure'] + 1)
    df_processed['HighRisk'] = ((df_processed['SeniorCitizen'] == 'Yes') & (df_processed['Contract'] == 'Month-to-month')).astype(int)
    object_cols = df_processed.select_dtypes(include=['object']).columns.intersection(required_features)
    df_processed[object_cols] = df_processed[object_cols].astype(str)
    missing_cols = set(required_features) - set(df_processed.columns)
    for col in missing_cols:
        df_processed[col] = 0 if col in numeric_cols else df_processed[categorical_cols[0]].mode()[0]
    return df_processed[required_features]

def predict_churn_for_user_input(model, preprocessor):
    st.markdown("## 🔮 Predict Churn for a New Customer")
    st.markdown("Choose how you'd like to predict churn: enter details manually for one customer or upload a file for multiple customers.")

    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        st.error(f"❌ Error loading dataset from {DATASET_PATH}: {str(e)}")
        return

    # Updated required features with new engineered ones
    required_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCost', 'HighRisk'
    ]
    st.info(f"📋 Dataset expects {len(required_features)} input features: {required_features}")

    categorical_options = {
        'gender': ['Male', 'Female'],
        'SeniorCitizen': ['No', 'Yes'],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['Yes', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    }

    prediction_type = st.radio("**Select Prediction Type**", ["Manual Input", "Batch Prediction"], help="Choose 'Manual Input' to enter details for one customer, or 'Batch Prediction' to upload a file for multiple customers.")

    if prediction_type == "Manual Input":
        st.markdown("### 📝 Manual Input: Customer Details")
        with st.form(key='user_input_form'):
            st.markdown("#### Customer Profile")
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", options=categorical_options['gender'])
                senior_citizen = st.selectbox("SeniorCitizen", options=categorical_options['SeniorCitizen'])
                partner = st.selectbox("Partner", options=categorical_options['Partner'])
            with col2:
                dependents = st.selectbox("Dependents", options=categorical_options['Dependents'])
                tenure = st.number_input("Tenure (months)", min_value=0, value=0)

            st.markdown("#### Services")
            col3, col4 = st.columns(2)
            with col3:
                phone_service = st.selectbox("Phone Service", options=categorical_options['PhoneService'])
                multiple_lines = st.selectbox("Multiple Lines", options=categorical_options['MultipleLines'])
                internet_service = st.selectbox("Internet Service", options=categorical_options['InternetService'])
                online_security = st.selectbox("Online Security", options=categorical_options['OnlineSecurity'])
                online_backup = st.selectbox("Online Backup", options=categorical_options['OnlineBackup'])
            with col4:
                device_protection = st.selectbox("Device Protection", options=categorical_options['DeviceProtection'])
                tech_support = st.selectbox("Tech Support", options=categorical_options['TechSupport'])
                streaming_tv = st.selectbox("Streaming TV", options=categorical_options['StreamingTV'])
                streaming_movies = st.selectbox("Streaming Movies", options=categorical_options['StreamingMovies'])

            st.markdown("#### Billing Information")
            col5, col6 = st.columns(2)
            with col5:
                contract = st.selectbox("Contract", options=categorical_options['Contract'])
                paperless_billing = st.selectbox("Paperless Billing", options=categorical_options['PaperlessBilling'])
                payment_method = st.selectbox("Payment Method", options=categorical_options['PaymentMethod'])
            with col6:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=0.01)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=0.01)

            submit_button = st.form_submit_button(label='🔍 Predict Churn')

        if submit_button:
            user_input = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            user_df = pd.DataFrame([user_input])
            try:
                user_df = preprocess_data(user_df, required_features)
                preprocessed_user_input = preprocessor.transform(user_df)
                if hasattr(preprocessed_user_input, 'toarray'):
                    preprocessed_user_input = preprocessed_user_input.toarray()

                with st.spinner("Predicting churn... ⏳"):
                    prediction = model.predict(preprocessed_user_input)
                    prediction_proba = model.predict_proba(preprocessed_user_input)[:, 1]

                st.markdown("#### 🎯 Prediction Result")
                if prediction[0] == 1:
                    st.error(f"🚨 Customer is **likely to churn** with a probability of {prediction_proba[0]:.2%}.")
                else:
                    st.success(f"🎉 Customer is **not likely to churn** with a probability of {1 - prediction_proba[0]:.2%} (churn probability: {prediction_proba[0]:.2%}).")

                st.markdown("#### 🔍 SHAP Explanation: Feature Contributions to the Prediction")
                with st.spinner("Computing SHAP values... ⏳"):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(preprocessed_user_input)
                    if len(shap_values.shape) == 1:
                        shap_values = shap_values.reshape(1, -1)

                    feature_names = preprocessor.get_feature_names_out()
                    feature_names = [name.split('__')[-1] for name in feature_names]
                    shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_values[0]})
                    shap_df['Absolute SHAP Value'] = shap_df['SHAP Value'].abs()
                    shap_df = shap_df.sort_values(by='Absolute SHAP Value', ascending=False)

                    st.markdown("#### 📋 Top Features Contributing to the Prediction")
                    st.write(shap_df.head(10))

                    fig = px.bar(shap_df.head(10), x='SHAP Value', y='Feature', orientation='h',
                                 title="Top 10 Features Contributing to Churn Prediction",
                                 color='SHAP Value', color_continuous_scale='Blues')
                    fig.update_layout(yaxis={'autorange': 'reversed'}, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### 📊 SHAP Force Plot")
                    shap.initjs()
                    plt.figure()
                    shap.force_plot(explainer.expected_value, shap_values[0], preprocessed_user_input[0],
                                   feature_names=feature_names, matplotlib=True, show=False)
                    st.pyplot(plt.gcf())

                report_content = {
                    "Prediction Result": f"Customer is {'likely' if prediction[0] == 1 else 'not likely'} to churn with a probability of {prediction_proba[0]:.2%}.",
                    "Top Features Contributing to the Prediction": shap_df.head(10)
                }
                pdf_buffer = generate_pdf_report(report_content, title="Churn Prediction Report")
                st.download_button(label="📥 Download Prediction Report (PDF)", data=pdf_buffer,
                                  file_name="churn_prediction_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

    elif prediction_type == "Batch Prediction":
        st.markdown("### 📂 Batch Prediction: Upload Customer Data File")
        st.markdown(f"**Instructions**: Upload an Excel file with these columns: {', '.join(required_features[:-2])}. The app will compute 'AvgMonthlyCost' and 'HighRisk'.")
        uploaded_file = st.file_uploader("Upload an Excel file with customer data", type=["xlsx", "xls"])
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_excel(uploaded_file)
                st.success(f"✅ Successfully loaded {df_uploaded.shape[0]} customer records.")
                required_columns = set(required_features[:-2])  # Exclude engineered features
                uploaded_columns = set(df_uploaded.columns)
                missing_columns = required_columns - uploaded_columns
                extra_columns = uploaded_columns - required_columns
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    return
                if extra_columns:
                    st.warning(f"⚠️ Extra columns found and will be ignored: {', '.join(extra_columns)}")
                    df_uploaded = df_uploaded.drop(columns=list(extra_columns))

                df_uploaded = preprocess_data(df_uploaded, required_features)
                preprocessed_data = preprocessor.transform(df_uploaded)
                if hasattr(preprocessed_data, 'toarray'):
                    preprocessed_data = preprocessed_data.toarray()

                with st.spinner("Predicting churn for all customers... ⏳"):
                    predictions = model.predict(preprocessed_data)
                    prediction_probs = model.predict_proba(preprocessed_data)[:, 1]

                with st.spinner("Computing SHAP values... ⏳"):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(preprocessed_data)

                feature_names = preprocessor.get_feature_names_out()
                feature_names = [name.split('__')[-1] for name in feature_names]
                results_df = pd.DataFrame({
                    'Customer ID': df_uploaded['customerID'] if 'customerID' in df_uploaded.columns else range(len(predictions)),
                    'Churn Prediction': ['Yes' if pred == 1 else 'No' for pred in predictions],
                    'Churn Probability': [f"{prob:.2%}" for prob in prediction_probs]
                })
                top_features = []
                for i in range(len(predictions)):
                    shap_vals = shap_values[i]
                    abs_shap = np.abs(shap_vals)
                    top_indices = np.argsort(abs_shap)[-3:][::-1]
                    top_feats = [feature_names[idx] for idx in top_indices]
                    top_features.append(", ".join(top_feats))
                results_df['Top Contributing Features'] = top_features

                st.markdown("#### 📋 Batch Prediction Results")
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button(label="📥 Download Prediction Results (CSV)", data=csv,
                                  file_name="churn_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Error processing uploaded file: {str(e)}")

    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"

def display_model_insights():
    st.markdown("## 🔍 Model Insights")
    st.markdown("Explore the precomputed insights of the churn prediction model (Steps 1-8).")
    
    report_content = {}
    
    with st.spinner("Loading Step 1..."):
        display_step_1()
        report_content["Step 1: Data Loading and Exploration"] = load_precomputed_data(DATA_HEAD_PATH)
    
    with st.spinner("Loading Step 2..."):
        display_step_2()
        report_content["Step 2: Data Preprocessing"] = "\n".join(load_precomputed_data(PREPROCESSING_LOG_PATH, file_type="lines") or [])
    
    with st.spinner("Loading Step 3..."):
        display_step_3()
        report_content["Step 3: Exploratory Data Analysis"] = "EDA plots are available in the app."
    
    with st.spinner("Loading Step 4..."):
        display_step_4()
        report_content["Step 4: Feature Engineering"] = load_precomputed_data(FEATURE_ENGINEERING_LOG_PATH, file_type="txt")
    
    with st.spinner("Loading Step 5..."):
        display_step_5()
        report_content["Step 5: Model Training"] = load_precomputed_data(TRAINING_LOG_PATH, file_type="txt")
    
    with st.spinner("Loading Step 6..."):
        display_step_6()
        report_content["Step 6: Model Evaluation"] = "\n".join(load_precomputed_data(EVALUATION_METRICS_PATH, file_type="lines") or [])
    
    with st.spinner("Loading Step 7..."):
        display_step_7()
        report_content["Step 7: SHAP Explainability"] = load_precomputed_data(FEATURE_IMPORTANCE_PATH)
    
    with st.spinner("Loading Step 8..."):
        display_step_8()
        report_content["Step 8: Business Recommendations"] = "\n".join([
            "Top features influencing customer churn:",
            str(load_precomputed_data(FEATURE_IMPORTANCE_PATH)),
            "Key Recommendations:",
            "\n".join([
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
            ])
        ])
    
    pdf_buffer = generate_pdf_report(report_content, title="Model Insights Report")
    st.download_button(
        label="📥 Download Model Insights Report (PDF)",
        data=pdf_buffer,
        file_name="model_insights_report.pdf",
        mime="application/pdf"
    )
    
    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"

def display_home_page():
    st.markdown('<div class="main-title">📡 Telecom Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Welcome to the Telecom Churn Prediction App! 🚀<br>This app uses a pre-trained XGBoost model to predict customer churn and provides insights into the model\'s workings. Choose an option below to get started.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🔍 Model Insights</h3>
            <p>Explore the precomputed insights of the churn prediction model, including data exploration, preprocessing, EDA, and more (Steps 1-8).</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Model Insights", key="model_insights_btn"):
            st.session_state.page = "model_insights"
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🔮 Predict Churn</h3>
            <p>Enter customer details to predict whether they are likely to churn and see which features contribute most to the prediction.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Predict Churn", key="predict_churn_btn"):
            st.session_state.page = "predict_churn"

def main():
    st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📡", layout="wide")
    
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
    }
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 18px;
        color: #4B5EAA;
        text-align: center;
        margin-bottom: 40px;
    }
    .sidebar .sidebar-content {
        background-color: #F0F2F6;
    }
    .card {
        background: linear-gradient(135deg, #E6F0FA 0%, #FFFFFF 100%);
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 20px;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .card h3 {
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    .card p {
        color: #666;
        font-size: 14px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #4B5EAA;
    }
    .stExpander {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Telecom Churn Prediction")
        st.markdown("---")
        
        st.markdown("#### About")
        st.markdown("This app helps telecom companies predict customer churn using machine learning. Navigate using the buttons on the Home page to explore model insights or predict churn for a new customer.")
        st.markdown("---")
        st.markdown("Developed by Kashish Ahuja")
        
        st.markdown("---")
        st.markdown("#### Feedback")
        with st.form(key="feedback_form"):
            feedback = st.text_area("We'd love to hear your feedback!", placeholder="Enter your suggestions or issues here...")
            submit_feedback = st.form_submit_button("Submit Feedback")
            if submit_feedback and feedback:
                st.success("Thank you for your feedback!")
    
    model, preprocessor = load_model_and_preprocessor()
    if model is None or preprocessor is None:
        return
    
    if st.session_state.page == "home":
        display_home_page()
    elif st.session_state.page == "model_insights":
        display_model_insights()
    elif st.session_state.page == "predict_churn":
        predict_churn_for_user_input(model, preprocessor)
    else:
        st.error("Invalid page state. Returning to Home.")
        st.session_state.page = "home"
        display_home_page()

if __name__ == "__main__":
    main()