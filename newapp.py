import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the model
xgb_model = joblib.load("xgb1.joblib.dat")

# Load the label encoder
label_encoder = joblib.load("../Harvestify-master/notebooks/label_encoder.joblib.dat")

def classify(answer):
    return f"The recommended crop for cultivation is: {label_encoder.inverse_transform(answer)}"

def main():

    st.markdown(
        """
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css" integrity="sha384-rvPz5Zt8ttfvi0va09ntZl5r1bjk7cP+5FkkTqBu5Jx2BC8/MF6n4QvL/0aIQomS" crossorigin="anonymous">
        </head>
        """,
        unsafe_allow_html=True
    )
    st.title("Crop Recommendation Application")
    st.subheader("Thinking what crop to grow?")
    image = Image.open('../Harvestify-master/1.JPG')
    st.image(image, use_column_width=True)
    # Header with background color
    st.markdown(
        """
        <style>
            .header {
                background-color: #4CAF50;
                padding: 15px;
                color: white;
                text-align: center;
                font-size: 24px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )    
    st.markdown('<h4 class="header"> Here\'s the solution!!!</h4>', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; background-color: #000045; color: #ece5f6'>Find The Most Suitable Crop</h1>", unsafe_allow_html=True)
    # User inputs with icons
    st.subheader("")
    st.markdown("<h6 style='text-align: center; background-color: #4CAF50; color: #ece5f6; padding: 10px;'>Select Environmental Factors</h6>", unsafe_allow_html=True)

    # Add icons to the buttons using HTML
    sn = st.slider('NITROGEN (N)', 0.0, 150.0, value=0.0)
    sp = st.slider('PHOSPHOROUS (P)', 0.0, 150.0, value=0.0)
    pk = st.slider('POTASSIUM (K)', 0.0, 210.0, value=0.0)
    pt = st.slider('TEMPERATURE ', 0.0, 50.0, value=0.0)
    phu = st.slider('HUMIDITY ', 0.0, 100.0, value=0.0)
    pPh = st.slider('Ph', 0.0, 14.0, value=0.0)
    pr = st.slider('RAINFALL ', 0.0, 300.0, value=0.0)

    # Prediction button
    if st.button("Get Crop Recommendation"):
        inputs = np.array([[sn, sp, pk, pt, phu, pPh, pr]])
        prediction = xgb_model.predict(inputs)
        st.success(classify(prediction))

    st.subheader("About")
    st.markdown(
        """
        <style>
            .about {
                background-color: #B0E57C;
                padding: 15px;
                text-align: justify;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Add a footer to your application
    st.markdown('<p class="about">This application uses XGBoost to predict the most suitable crop for cultivation based on environmental factors. <br /> Why XGBoost ? <br /> Because it gives the best accuracy!! </p>', unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            .footer {
                background-color: #4CAF50;
                padding: 15px;
                color: white;
                text-align: center;
                font-size: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<p class="footer">Created by TEAM 13</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
