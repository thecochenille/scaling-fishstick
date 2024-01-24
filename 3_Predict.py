import streamlit as st
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import shap
import pandas as pd

# Load the pre-trained Spark model
spark = SparkSession.builder.appName("ChurnPredictionApp").getOrCreate()
rf_model = PipelineModel.load("rf_model")
test_data = spark.read.parquet("test_data")
test_predictions = spark.read.parquet("test_predictions")

# Get feature names and importance scores
feature_names = test_data.columns  # replace with your actual feature names
feature_importance = rf_model.stages[-1].featureImportances

# Streamlit App
st.title("Churn Prediction App")
st.write(test_data.head())
st.write(test_predictions.head())



st.write(feature_importance)






# Convert Spark DataFrame to Pandas DataFrame (if needed)
test_data_pd = test_data.toPandas()

# Explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(spark_model)
shap_values = explainer.shap_values(test_data_pd)

# Plot summary plot for feature importance
shap.summary_plot(shap_values, test_data_pd)


# User Input
st.sidebar.header("User Input")

# Create input components based on feature importance
for feature, importance in zip(feature_names, feature_importance):
    # You can customize the input type based on the feature type
    user_input = st.sidebar.number_input(f"Enter {feature}:", min_value=0.0)

# Perform predictions based on user input
if st.sidebar.button("Predict Churn"):
    # Create a DataFrame with user input
    user_data = pd.DataFrame({feature: [user_input] for feature, user_input in zip(feature_names, user_input)})
    spark_user_data = spark.createDataFrame(user_data)

    # Make predictions
    predictions = rf_model.transform(spark_user_data)

    # Display Churn Prediction
    st.subheader("Churn Prediction:")
    st.write(predictions.select("prediction").collect()[0][0])
