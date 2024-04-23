import numpy as np
import pandas as pd
import datetime
import pickle
import streamlit as st
from sklearn import preprocessing

# Load necessary resources
model = pickle.load(open('./CHLA_Prediction/CHLA_Deployment/random_forest.pkl', 'rb'))
encoder_dict = pickle.load(open('./CHLA_Prediction/CHLA_Deployment/encoder_V2.pkl', 'rb'))
dataset = pd.read_csv('./CHLA_Prediction/CHLA_Deployment/CHLA_clean_data_2024_Appointments.csv')
dataset.columns = dataset.columns.str.upper()
dataset['APPT_DATE'] = pd.to_datetime(dataset['APPT_DATE'], errors='coerce')

def main():
    st.title("Appointment Showup Prediction")
    clinics = dataset['CLINIC'].dropna().unique()
    clinic = st.selectbox("Select a Clinic", options=clinics)
    appt_date_range = st.date_input("Select Appointment Date Range", value=[dataset['APPT_DATE'].min(), dataset['APPT_DATE'].max()])

    if st.button("Predict"):
        filtered_data = dataset[
            (dataset['CLINIC'] == clinic) &
            (dataset['APPT_DATE'].dt.date >= appt_date_range[0]) &
            (dataset['APPT_DATE'].dt.date <= appt_date_range[1])
        ]
        
        if not filtered_data.empty:
            predictions = model.predict(filtered_data)
            filtered_data['PREDICTION'] = predictions[:len(filtered_data)]  # Ensure predictions align with filtered data

            st.write(filtered_data)
        else:
            st.error("No data available for the selected clinic and date range.")

if __name__ == '__main__':
    main()

