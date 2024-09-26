import streamlit as st
import pandas as pd
import pickle

# Load the trained Lasso model
filename_lasso = 'lasso_regression_model.sav'
lasso_model = pickle.load(open(filename_lasso, 'rb'))

# Create the Streamlit app
st.title("Monthly Revenue Prediction App")

# Get user input for features
st.header("Enter store features:")
website_traffic = st.number_input("Website Traffic", min_value=0)
avg_order_value = st.number_input("Average Order Value", min_value=0.0)
customer_acquisition_cost = st.number_input("Customer Acquisition Cost", min_value=0.0)
marketing_spend = st.number_input("Marketing Spend", min_value=0.0)
customer_lifetime_value = st.number_input("Customer Lifetime Value", min_value=0.0)
social_media_engagement = st.number_input("Social Media Engagement", min_value=0)
conversion_rate = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0)
average_order_processing_time = st.number_input("Average Order Processing Time (days)", min_value=0)

# Create a button to predict
if st.button("Predict Monthly Revenue"):
  # Prepare input data for prediction
  input_data = pd.DataFrame({
      'website_traffic': [website_traffic],
      'avg_order_value': [avg_order_value],
      'customer_acquisition_cost': [customer_acquisition_cost],
      'marketing_spend': [marketing_spend],
      'customer_lifetime_value': [customer_lifetime_value],
      'social_media_engagement': [social_media_engagement],
      'conversion_rate': [conversion_rate],
      'average_order_processing_time': [average_order_processing_time]
  })

  # Make the prediction using the Lasso model
  predicted_revenue = lasso_model.predict(input_data)[0]

  # Display the prediction
  st.header("Predicted Monthly Revenue:")
  st.write(f"${predicted_revenue:.2f}")

