BI-2.0-AI-ML-Project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Business Intelligence 2.0")
st.subheader("AI & ML for Real-Time Decision Making")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.write(data.head())

    # Preprocessing
    data = data.dropna()
    data = data.drop_duplicates()

    # Feature selection
    X = data[['Month']]
    y = data['Sales']

    # Model
    model = LinearRegression()
    model.fit(X, y)

    # Prediction
    predictions = model.predict(X)

    # Visualization
    st.write("### Visualization")
    fig, ax = plt.subplots()
    ax.plot(X, y, label="Actual")
    ax.plot(X, predictions, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    st.success("Prediction completed successfully!")
