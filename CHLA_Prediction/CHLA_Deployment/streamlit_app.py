import streamlit as st
import pandas as pd
import datetime
import pickle
from sklearn import preprocessing

# Load the model
model = pickle.load(open('./CHLA_Prediction/CHLA_Deployment/random_forest.pkl', 'rb'))

# Load the encoder dictionary
encoder_dict = pickle.load(open('./CHLA_Prediction/CHLA_Deployment/encoder_V2.pkl', 'rb'))

# Load the dataset
data_path = './CHLA_Prediction/CHLA_Deployment/CHLA_clean_data_2024_Appointments.csv'
dataset = pd.read_csv(data_path)
dataset.columns = dataset.columns.str.lower()
dataset['appt_date'] = pd.to_datetime(dataset['appt_date'], errors='coerce')

# Function to process features, including dates
def process_features(data, encoder_dict, date_columns=None):
    if date_columns is None:
        date_columns = []
    for column in data.columns:
        if column in date_columns:
            # Extract date features if the column is a date
            data[f'{column}_year'] = data[column].dt.year
            data[f'{column}_month'] = data[column].dt.month
            data[f'{column}_day'] = data[column].dt.day
        elif column in encoder_dict:
            # Encode categorical features
            le = preprocessing.LabelEncoder()
            le.classes_ = encoder_dict[column]
            data[column] = data[column].apply(lambda x: x if x in le.classes_ else 'Unknown')
            data[column] = le.transform(data[column])
    # Optionally drop original date columns if they are not needed anymore
    data.drop(columns=date_columns, errors='ignore', inplace=True)
    return data

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Appointment Showup Prediction")
    st.title("Appointment Showup Prediction")

    # User inputs for clinic and range of appointment dates
    clinics = dataset['clinic'].dropna().unique()
    clinic = st.selectbox("Select a Clinic", options=clinics)
    appt_date_range = st.date_input("Select Appointment Date Range", value=[dataset['appt_date'].min(), dataset['appt_date'].max()], min_value=dataset['appt_date'].min(), max_value=dataset['appt_date'].max())

    if st.button("Predict"):
        # Filter the dataset based on the selected clinic and appointment date range
        filtered_data = dataset[(dataset['clinic'] == clinic) &
        (dataset['appt_date'].dt.date >= appt_date_range[0]) &
        (dataset['appt_date'].dt.date <= appt_date_range[1])]

        if not filtered_data.empty:
            # Process features for prediction, specifying which columns are dates
            processed_data = process_features(filtered_data.copy(), encoder_dict, date_columns=['appt_date'])
            # Prepare features for prediction
            predictions = model.predict(processed_data.drop(['is_noshow'], axis=1, errors='ignore'))

            # Append the predictions to the filtered DataFrame
            filtered_data['Prediction'] = predictions

            if 'is_noshow' in filtered_data.columns:
                accuracy = (filtered_data['Prediction'] == filtered_data['is_noshow']).mean() * 100
                st.success(f"Prediction Accuracy: {accuracy:.2f}%")
            
            # Display the complete DataFrame with predictions
            st.write(filtered_data)
        else:
            st.error("No data available for the selected clinic and date range.")

if __name__ == '__main__':
    main()
