import streamlit as st
import pandas as pd
import pickle
import numpy as np
import io
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

def display_ui():
    # แบ่งส่วนหน้าจอออกเป็น 2 คอลัมน์
    col1, col2 = st.columns([2, 3])

    # ส่วนของ col1 (พื้นที่สีส้ม)
    with col1:
        with st.container():
            st.markdown('''
                <div style="background-color:orange; padding:10px; margin-top:20px;">
                    <h3 style="color:white; text-align: center;">Generate Data</h3>
                </div>
            ''', unsafe_allow_html=True)
            dataset1 = st.file_uploader("Upload Generate Dataset 1 (CSV)", type="csv", key="dataset1")

            st.markdown('''
                <div style="background-color:orange; padding:10px; margin-top:20px;">
                    <h3 style="color:white; text-align: center;">Weather Data</h3>
                </div>
            ''', unsafe_allow_html=True)
            dataset2 = st.file_uploader("Upload Weather Dataset 2 (CSV)", type="csv", key="dataset2")

    # คอลัมน์ที่ 2 (พื้นที่แสดงข้อมูล - สีฟ้า)
    with col2:
        with st.container():
            st.markdown('''
                <div style="background-color:#ADD8E6; padding:20px; height:60vh; display: flex; flex-direction: column; justify-content: flex-start;">
                    <h1 style="color:white; text-align: center; width: 100%; margin-top: 0;">Solar Power Prediction</h1>
                </div>
            ''', unsafe_allow_html=True)

    return dataset1, dataset2

def process_datasets(dataset1, dataset2, model_choice):
    if dataset1 is not None and dataset2 is not None:
        st.write("Dataset 1 and Dataset 2 uploaded successfully!")

        # อ่านข้อมูลทั้งสองไฟล์
        df_Generation_Data = pd.read_csv(io.StringIO(dataset1.getvalue().decode('utf-8')))
        df_Weather_Data = pd.read_csv(io.StringIO(dataset2.getvalue().decode('utf-8')))

        # แปลง DATE_TIME ทั้งสอง DataFrame
        df_Generation_Data['DATE_TIME'] = pd.to_datetime(df_Generation_Data['DATE_TIME'], errors='coerce')
        df_Weather_Data['DATE_TIME'] = pd.to_datetime(df_Weather_Data['DATE_TIME'], errors='coerce')

        # ทำการ merge ข้อมูลสอง DataFrame ตามคอลัมน์ DATE_TIME
        merged_df = pd.merge(df_Generation_Data, df_Weather_Data, on='DATE_TIME', how='inner')
        st.dataframe(merged_df)

        merged_df['DATE_TIME'] = pd.to_datetime(merged_df['DATE_TIME'])
        merged_df['HOUR'] = merged_df['DATE_TIME'].dt.hour
        merged_df['DAY'] = merged_df['DATE_TIME'].dt.day
        merged_df['MONTH'] = merged_df['DATE_TIME'].dt.month
        y = merged_df[['DC_POWER', 'AC_POWER', 'TOTAL_YIELD']]
        merged_df = merged_df.drop(['DATE_TIME', 'PLANT_ID_x', 'SOURCE_KEY_x', 'PLANT_ID_y', 'SOURCE_KEY_y'], axis=1)
        merged_df = merged_df.drop(['DC_POWER', 'AC_POWER', 'TOTAL_YIELD'], axis=1)
        X = merged_df
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == 'XGBoost':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Standardize the features
                ('xgb', XGBRegressor(n_estimators=100, random_state=42))  # XGBoost Regressor
            ])
        elif model_choice == 'RandomForest':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Standardize the features
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42))  # RandomForest Regressor
            ])
        elif model_choice == 'LinearRegression':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Standardize the features
                ('lr', LinearRegression())  # Linear Regression model
            ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        st.write(y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f'MSE: {mse}')

        # Plot DC_POWER comparison
        plt.figure(figsize=(14, 12))
        plt.subplot(3, 1, 1)  # Subplot 1: DC_POWER
        plt.plot(merged_df.index[-len(y_test):], y_test['DC_POWER'], color='blue', label='Actual DC Power')
        plt.plot(merged_df.index[-len(y_test):], y_pred[:, 0], color='orange', linestyle='dashed', label='Predicted DC Power')
        plt.title(f'{model_choice} - DC Power: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('DC Power')
        plt.legend()
        st.pyplot(plt)

# เรียกใช้ฟังก์ชัน UI
dataset1, dataset2 = display_ui()

# เพิ่มปุ่มเลือกโมเดล
model_choice = st.selectbox('Select Model', ('XGBoost', 'RandomForest', 'LinearRegression'))

# ปุ่มสำหรับกดเพื่อเรียกใช้โมเดล
if st.button('Run Model'):
    process_datasets(dataset1, dataset2, model_choice)
