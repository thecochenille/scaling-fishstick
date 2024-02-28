#streamlit app showing transformed data with some insights

import numpy as np
import pandas as pd
from scipy import stats

import streamlit as st




def user_input_features():
	gender = st.sidebar.selectbox('gender',('Male','Female'))


#dataset
df = pd.read_csv("user_data.csv")


# Streamlit app
st.title("Sparkify user behavior analysis")
st.write("Welcome to the Sparkify Churn Prediction App! \
	Sparkify is a fictional music streaming service, and this app \
	is designed to predict customer churn, helping us understand \
	and anticipate when users might consider canceling their subscription.\
	Leveraging Leveraging advanced machine learning algorithms, the app \
	analyzes user engagement patterns, listening habits, and other \
	relevant features to identify potential churners. The goal is to empower \
	Sparkify with insights that enable proactive and targeted retention \
	strategies. Explore the predictions and insights provided by our \
	churn prediction model to enhance user retention strategies and \
	ensure a seamless musical experience for all our users")
st.write(df) 


# SIDEBAR
st.sidebar.header("Explore the dataset")

st.sidebar.write("")

st.sidebar.header("Type of customers")

