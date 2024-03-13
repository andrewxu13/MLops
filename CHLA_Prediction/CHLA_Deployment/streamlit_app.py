import logging

# Configure logging
logging.basicConfig(filename='./app.log', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


import streamlit as st
import pandas as pd
import datetime
import pickle

model = pickle.load(open('./random_forest.pkl', 'rb'))

today = datetime.datetime.today()
date_string = datetime.datetime(2000, 1, 1)




def main():
    st.set_page_config(page_title="CHIA Prediction")
    st.title("Patient Showup Prediction: ")

    appt_date = st.date_input(label = "Appointment Date", value = today, min_value= date_string)
    book_date = st.date_input(label = "Booking Date", value = today, min_value= date_string)
    lead_time = st.text_input("Lead time", 0)
    total_no_show = st.text_input("How many times this patient hasn't show up in the past?",0)
    total_success = st.text_input("How many times this patient has show up for their appointment?",0)
    total_cancel = st.text_input("How many times this patient has cancel for their appointment?",0)
    total_reschedule = st.text_input("How many times this patient has reschedule for their appointment?",0)

    if st.button("Predict"):
        data = {
        'appt_date': appt_date,
        'book_date': book_date,
        'lead_time': int(lead_time),
        'total_no_show': int(total_no_show),
        'total_success': int(total_success),
        'total_cancel': int(total_cancel),
        'total_reschedule': int(total_reschedule)}
        
    

        predict_df = pd.DataFrame([data])

        predict_df['appt_date'] = pd.to_datetime(predict_df['appt_date'])
        predict_df['book_date'] = pd.to_datetime(predict_df['book_date'])

        predict_df['NUM_OF_MONTH'] = predict_df['appt_date'].dt.month

        # Now, all your features should be numerical, and you can attempt prediction
        features_list = predict_df.drop(['appt_date','book_date' ],axis = 1).values
        prediction = model.predict(features_list)
        
        if prediction == 1:
            text = "Patient will not show up"
        else:
            text = "Patient will show up"

        st.success(text)
        
# Refactor to a function for unit testing
def make_prediction(data):
    predict_df = pd.DataFrame([data])
    predict_df['appt_date'] = pd.to_datetime(predict_df['appt_date'])
    predict_df['book_date'] = pd.to_datetime(predict_df['book_date'])
    predict_df['NUM_OF_MONTH'] = predict_df['appt_date'].dt.month
    features_list = predict_df.drop(['appt_date', 'book_date'], axis=1).values
    prediction = model.predict(features_list)
    return prediction


if __name__=='__main__':
    main()
