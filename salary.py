import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and dataframe
salary_pipe = pickle.load(open('salary_pipe.pkl', 'rb'))
df_salary = pickle.load(open('salary_df.pkl', 'rb'))

st.title("Data Science Salary Prediction")

# User Input
a = st.number_input('Input Year')
b = st.selectbox('Experience Level', df_salary['experience_level'].unique())
c = st.selectbox('Employment Type', df_salary['employment_type'].unique())
d = st.selectbox('Job Title', df_salary['job_title'].unique())
e = st.number_input('Salary')
f = st.selectbox('Salary Currency', df_salary['salary_currency'].unique())
g = st.selectbox('Employee Residence', df_salary['employee_residence'].unique())
h = st.selectbox('Remote Ratio', df_salary['remote_ratio'].unique())
i = st.selectbox('Company Location', df_salary['company_location'].unique())
j = st.selectbox('Company Size', df_salary['company_size'].unique())

# Add a "Predict" button
if st.button("Predict"):
    # Prepare user input as a DataFrame
    user_input = pd.DataFrame({
        'work_year': [a],
        'experience_level': [b],
        'employment_type': [c],
        'job_title': [d],
        'salary': [e],
        'salary_currency': [f],
        'employee_residence': [g],
        'remote_ratio': [h],
        'company_location': [i],
        'company_size': [j]
    })

    # Make Prediction
    predicted_salary = salary_pipe.predict(user_input)

    # Display the result with a larger and bolder format
    st.subheader("Predicted Salary:")
    st.markdown(f"<p style='font-size:30px; font-weight: bold;'>${predicted_salary[0]:,.2f} 💵</p>", unsafe_allow_html=True)