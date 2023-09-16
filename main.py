import streamlit as st
import pandas as pd
import numpy as np
import sklearn
print(sklearn.__version__)
#load the model from disk
import joblib
model = joblib.load(r"./model.sav")

#Import python scripts
from preprocessing import preprocess

def main():
    #Setting Application title
    st.title('Customer Churn Predictor')

    st.markdown("<h3></h3>", unsafe_allow_html=True)


    st.info("Input data below")
    #Based on our optimal features selection
    st.subheader("Demographic data")
    gender = st.selectbox('Gender:', ('Male', 'Female'))
    loc = st.selectbox('Location:', ('Chicago', 'Houston', 'Los Angeles', 'Miami', 'New York'))
    st.subheader("Numeric data")
    age = st.number_input('Age of the Customer:', min_value=0, max_value=120, value=0)
    length_of_subscription = st.number_input('Subscription length in months:',min_value=0, max_value=10000, value=0)
    monthly_bill = st.number_input('Monthly Bill:', min_value=0, max_value=120000, value=0)
    total_usage_gb = st.number_input('Total Usage in GB:', min_value=0, max_value=1200, value=0)
    data = {
            'Gender': gender,
            'Location': loc,
            'Age': age,
            'Subscription_Length_Months': length_of_subscription,
            'Monthly_Bill': monthly_bill,
            'Total_Usage_GB': total_usage_gb,
            }
    features_df = pd.DataFrame.from_dict([data])
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.write('Overview of input is shown below')
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.dataframe(features_df)
    #Preprocess inputs
    preprocess_df = preprocess(features_df, 'Online')

    prediction = model.predict(preprocess_df)

    if st.button('Predict'):
        if prediction == 1:
            st.warning('Yes, the customer will terminate the service.')
        else:
            st.success('No, the customer is happy with Telco Services.')




if __name__ == '__main__':
        main()
