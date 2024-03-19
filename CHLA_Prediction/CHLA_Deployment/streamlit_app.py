import logging



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Add file handler
file_handler = logging.FileHandler('./app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)



import streamlit as st
import pandas as pd
import datetime
import sklearn
import pickle

model = pickle.load(open('./CHLA_Prediction/CHLA_Deployment/random_forest.pkl', 'rb'))

today = datetime.datetime.today()
date_string = datetime.datetime(2000, 1, 1)




import pandas as pd
import streamlit as st

# Function to make prediction
def make_prediction(data):
    predict_df = pd.DataFrame([data])
    predict_df['appt_date'] = pd.to_datetime(predict_df['appt_date'])
    predict_df['book_date'] = pd.to_datetime(predict_df['book_date'])
    predict_df['NUM_OF_MONTH'] = predict_df['appt_date'].dt.month
    features_list = predict_df.drop(['appt_date', 'book_date'], axis=1).values
    prediction = model.predict(features_list)
    return prediction

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Appointment Showup Prediction")
    st.title("Appointment Showup Prediction")

    appt_date = st.date_input(label="Appointment Date", value=today, min_value=date_string)
    book_date = st.date_input(label="Booking Date", value=today, min_value=date_string)
    lead_time = st.text_input("Lead time (in days)", 0)
    total_no_show = st.text_input("Total number of previous no-shows", 0)
    total_success = st.text_input("Total number of previous successful appointments", 0)
    total_cancel = st.text_input("Total number of previous cancellations", 0)
    total_reschedule = st.text_input("Total number of previous rescheduled appointments", 0)

    if st.button("Predict"):
        data = {
            'appt_date': appt_date,
            'book_date': book_date,
            'lead_time': int(lead_time),
            'total_no_show': int(total_no_show),
            'total_success': int(total_success),
            'total_cancel': int(total_cancel),
            'total_reschedule': int(total_reschedule)
        }

        prediction = make_prediction(data)
        
        if prediction == 1:
            text = "Prediction: Patient is likely to miss the appointment."
        else:
            text = "Prediction: Patient is likely to attend the appointment."

        st.success(text)

# Entry point for running the Streamlit app
if __name__ == '__main__':
    main()
