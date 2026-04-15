# SK-KALIMULLA
Business Intelligence 2.0: The Role Of Artificial Intelligence and Machine Learning in Real Time Decision Making
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Business Intelligence 2.0 Dashboard")

uploaded_file = st.file_uploader("Upload CSV File")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", data)

    X = data[['Month']]
    y = data['Sales']

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    data['Predicted Sales'] = predictions

    st.write("Predictions", data)

    st.line_chart(data[['Sales', 'Predicted Sales']])
