import numpy as np
import pandas as pd
import datetime
import pickle
import streamlit as st
from sklearn import preprocessing

# Load necessary resources
MODEL = pickle.load(open('./CHLA_Prediction/CHLA_Deployment/random_forest.pkl', 'rb'))
ENCODER_DICT = pickle.load(open('./CHLA_Prediction/CHLA_Deployment/encoder_V2.pkl', 'rb'))
DATASET = pd.read_csv('./CHLA_Prediction/CHLA_Deployment/CHLA_clean_data_2024_Appointments.csv')
DATASET.columns = DATASET.columns.str.upper()
DATASET['APPT_DATE'] = pd.to_datetime(DATASET['APPT_DATE'], errors='coerce')

def process_features(DATA, ENCODER_DICT):
    # Columns that should be dropped if they contain unseen categories
    DROPPABLE_COLUMNS = []
    for COLUMN in ENCODER_DICT:
        if COLUMN in DATA.columns:
            LE = preprocessing.LabelEncoder()
            LE.classes_ = ENCODER_DICT[COLUMN]  # Load previously fitted classes

            # Check if all current data points are in the known classes
            if not set(DATA[COLUMN].unique()).issubset(set(LE.classes_)):
                # If unseen values are found, mark the column for exclusion
                DROPPABLE_COLUMNS.append(COLUMN)
            else:
                # Transform the data as all categories are seen
                DATA[COLUMN] = LE.transform(DATA[COLUMN])

    # Drop columns with unseen categories
    DATA.drop(columns=DROPPABLE_COLUMNS, inplace=True, errors='ignore')
    return DATA

def main():
    st.title("Appointment Showup Prediction")
    CLINICS = DATASET['CLINIC'].dropna().unique()
    CLINIC = st.selectbox("Select a Clinic", options=CLINICS)
    APPT_DATE_RANGE = st.date_input("Select Appointment Date Range", value=[DATASET['APPT_DATE'].min(), DATASET['APPT_DATE'].max()])

    if st.button("Predict"):
        FILTERED_DATA = DATASET[
            (DATASET['CLINIC'] == CLINIC) &
            (DATASET['APPT_DATE'].dt.date >= APPT_DATE_RANGE[0]) &
            (DATASET['APPT_DATE'].dt.date <= APPT_DATE_RANGE[1])
        ]

        if not FILTERED_DATA.empty:
            PROCESSED_DATA = process_features(FILTERED_DATA.copy(), ENCODER_DICT)
            PREDICTIONS = MODEL.predict(PROCESSED_DATA)
            FILTERED_DATA['PREDICTION'] = PREDICTIONS[:len(FILTERED_DATA)]  # Ensure predictions align with filtered data

            st.write(FILTERED_DATA)
        else:
            st.error("No data available for the selected clinic and date range.")

if __name__ == '__main__':
    main()
