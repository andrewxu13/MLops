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
dataset['APPT_DATE'] = pd.to_datetime(dataset['APPT_DATE'], errors='coerce')

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
    clinics = dataset['CLINIC'].dropna().unique()
    clinic = st.selectbox("Select a Clinic", options=clinics)
    appt_date_range = st.date_input("Select Appointment Date Range", value=[dataset['APPT_DATE'].min(), dataset['APPT_DATE'].max()], min_value=dataset['APPT_DATE'].min(), max_value=dataset['APPT_DATE'].max())

    if st.button("Predict"):
        # Filter the dataset based on the selected clinic and appointment date range
        filtered_data = dataset[(dataset['CLINIC'] == clinic) &
        (dataset['APPT_DATE'].dt.date >= appt_date_range[0]) &
        (dataset['APPT_DATE'].dt.date <= appt_date_range[1])]

        if not filtered_data.empty:
            # Process features for prediction, specifying which columns are dates
            processed_data = process_features(filtered_data.copy(), encoder_dict, date_columns=['APPT_DATE'])
            # Prepare features for prediction
            predictions = model.predict(processed_data.drop(['IS_NOSHOW'], axis=1, errors='ignore'))

            # Append the predictions to the filtered DataFrame
            filtered_data['Prediction'] = predictions

            if 'is_noshow' in filtered_data.columns:
                accuracy = (filtered_data['Prediction'] == filtered_data['IS_NOSHOW']).mean() * 100
                st.success(f"Prediction Accuracy: {accuracy:.2f}%")
            
            # Display the complete DataFrame with predictions
            st.write(filtered_data)
        else:
            st.error("No data available for the selected clinic and date range.")

if __name__ == '__main__':
    main()
