import streamlit as st
st.set_page_config(page_title="Somalia Electricity Dashboard (Real Data)", layout="centered")

import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load real dataset
df = pd.read_csv("somalia_electricity_data_real.csv")

# Load model if exists
try:
    model = joblib.load("electricity_rf_model_real.pkl")
    model_loaded = True
except:
    model_loaded = False
    st.warning("⚠️ Trained model not found. Please train and export the model as 'electricity_rf_model_real.pkl'.")

st.title("🔌 Somalia Electricity Price Dashboard (Real Data)")
st.markdown("Explore real data and predict electricity prices for Somali regions.")

# Show data
st.subheader("📄 Dataset Preview")
st.dataframe(df)

# Visualize pricing
st.subheader("📊 Electricity Price by Provider Type")
fig, ax = plt.subplots()
sns.boxplot(x="Provider_Type", y="Electricity_kWh_Price_USD", data=df, ax=ax)
plt.xticks(rotation=15)
st.pyplot(fig)

# Input fields
st.subheader("🔮 Predict New Price")
diesel = st.slider("Diesel Price (USD/L)", 1.0, 1.5, 1.25, 0.01)
grid = st.slider("Grid Coverage (%)", 0, 100, 40)
income = st.slider("Avg Household Income (USD)", 100, 500, 250)
infra = st.slider("Infrastructure Index", 1.0, 5.0, 2.5, 0.1)

if st.button("Predict Price"):
    if model_loaded:
        input_df = pd.DataFrame({
            'Diesel_Price_USD_per_L': [diesel],
            'Grid_Coverage_Percent': [grid],
            'Avg_Household_Income_USD': [income],
            'Infrastructure_Quality_Index': [infra]
        })
        prediction = model.predict(input_df)[0]
        st.success(f"💡 Predicted Electricity Price: **${prediction:.2f} per kWh**")
    else:
        st.error("Model not loaded. Please make sure 'electricity_rf_model_real.pkl' is available.")
