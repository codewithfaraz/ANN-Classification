import tensorflow as tf
import numpy as np
import streamlit as st
import pickle
import pandas as pd

#load the trained model
model = tf.keras.models.load_model('model.h5')
st.title("Churn Probability Prediction")
#load the encoders
with open("label_encoder_gender.pkl", "rb") as f:
    le = pickle.load(f)

with open("onhot_encoder_geo.pkl", "rb") as f:
    ohe = pickle.load(f)
#Load the scaler
with open("scaler.pkl",'rb') as f:
    scaler = pickle.load(f)

#User input
geogrphy = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox("Gender",le.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#Prepare the nput data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
}

#One hot encode the geography
geo = ohe.transform([[geogrphy]]).toarray()
geo_encoded_df = pd.DataFrame(geo, columns=ohe.get_feature_names_out())

input_data_df = pd.DataFrame(input_data)
input_data_df = pd.concat([input_data_df, geo_encoded_df], axis=1)
scaled_data = scaler.transform(input_data_df)
print(scaled_data)


#Predict
prediction = model.predict(scaled_data)
prediction = prediction[0][0]
st.write("Prediction: ", prediction)
if prediction > 0.5:
    st.write("Customer will exit")
else:
    st.write("Customer will not exit")