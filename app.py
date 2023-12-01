#streamlit app showing transformed data with some insights

import numpy as np
import pandas as pd
from scipy import stats

import streamlit as st


def user_input_features():
	gender = st.sidebar.selectbox('gender',('Male','Female'))


# Streamlit app
st.title("Sparkify user behavior analysis")


# Add a sidebar for user input
st.sidebar.header("Written by Isabelle Vea")

st.sidebar.write(" 2. Set a large sample number such as 50 or 100 and look at the histogram, does it look normal now?")



st.sidebar.header("Type of customers")

