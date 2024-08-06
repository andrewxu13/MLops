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
        # Filter data for prediction
        prediction_data = dataset[
            (dataset['CLINIC'] == clinic) &
            (dataset['APPT_DATE'].dt.date >= appt_date_range[0]) &
            (dataset['APPT_DATE'].dt.date <= appt_date_range[1])
        ]

        # Prepare data for prediction (make sure to use the correct column names as per your model's requirements)
        processed_data = prediction_data[['LEAD_TIME', 'TOTAL_NUMBER_OF_NOSHOW', 'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT',
                                          'TOTAL_NUMBER_OF_CANCELLATIONS', 'TOTAL_NUMBER_OF_RESCHEDULED', 'NUM_OF_MONTH']]

        if not processed_data.empty:
            probabilities = model.predict_proba(processed_data)
            predictions = np.argmax(probabilities, axis=1)  # Getting the class with the highest probability
            prediction_data['PREDICTION'] = predictions
            prediction_data['PROBABILITY'] = probabilities[:, 1]  # Assuming class 1 is the positive class

            # Filter columns for display after predictions
            display_data = prediction_data[['MRN', 'APPT_DATE', 'BOOK_DATE', 'CLINIC', 'IS_NOSHOW', 'PREDICTION']]
            display_data['PROBABILITY_BAR'] = prediction_data['PROBABILITY'].apply(lambda x: st.progress(x))

            st.write(display_data)
        else:
            st.error("No data available for the selected clinic and date range.")

if __name__ == '__main__':
    main()
