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
dataset.columns = dataset.columns.str.lower()
dataset['appt_date'] = pd.to_datetime(dataset['appt_date'], errors='coerce')

def process_features(data, encoder_dict):
    # Columns that should be dropped if they contain unseen categories
    droppable_columns = []
    for column in encoder_dict:
        if column in data.columns:
            le = preprocessing.LabelEncoder()
            le.classes_ = encoder_dict[column]  # Load previously fitted classes

            # Check if all current data points are in the known classes
            if not set(data[column].unique()).issubset(set(le.classes_)):
                # If unseen values are found, mark the column for exclusion
                droppable_columns.append(column)
            else:
                # Transform the data as all categories are seen
                data[column] = le.transform(data[column])

    # Drop columns with unseen categories
    data.drop(columns=droppable_columns, inplace=True, errors='ignore')
    return data

def main():
    st.title("Appointment Showup Prediction")
    clinics = dataset['clinic'].dropna().unique()
    clinic = st.selectbox("Select a Clinic", options=clinics)
    appt_date_range = st.date_input("Select Appointment Date Range", value=[dataset['appt_date'].min(), dataset['appt_date'].max()])

    if st.button("Predict"):
        filtered_data = dataset[
            (dataset['clinic'] == clinic) &
            (dataset['appt_date'].dt.date >= appt_date_range[0]) &
            (dataset['appt_date'].dt.date <= appt_date_range[1])
        ]

        if not filtered_data.empty:
            processed_data = process_features(filtered_data.copy(), encoder_dict)
            predictions = model.predict(processed_data)
            filtered_data['Prediction'] = predictions[:len(filtered_data)]  # Ensure predictions align with filtered data

            st.write(filtered_data)
        else:
            st.error("No data available for the selected clinic and date range.")

if __name__ == '__main__':
    main()
