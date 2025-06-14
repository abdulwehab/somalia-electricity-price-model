# Somalia Electricity Price Prediction

This project models and predicts electricity prices in Somali regions using real-world inspired data. It includes a trained machine learning model and an interactive Streamlit dashboard.

---

## 📊 Dataset

The dataset includes 10 major Somali regions with the following features:

- Region
- Provider Type (e.g., Private Generator, Public Utility)
- Population
- Diesel Price (USD per liter)
- Grid Coverage (%)
- Solar Usage Level
- Average Household Income (USD)
- Infrastructure Quality Index
- Electricity Price (USD/kWh) — target variable

Data sources include:
- Somalia Ministry of Energy & Water Resources
- World Bank Open Data
- African Development Bank country reports
- USAID/Power Africa

---

## ⚙️ Machine Learning Models

Two models were trained and evaluated:

- **Linear Regression**
- **Random Forest Regressor** ✅ (Best performing)

**Performance:**
- R² Score ≈ 0.84
- MAE ≈ 0.12
- RMSE ≈ 0.17

The trained model was saved to a `.pkl` file for reuse.

---

## 🌐 Streamlit Dashboard

A user-friendly Streamlit dashboard was built to allow:
- Viewing region-wise electricity prices
- Visualizing price differences by provider type
- Inputting new values to get predicted prices

### Run the dashboard locally:

```bash
streamlit run streamlit_app_real.py
```

Make sure these files are in the same folder:
- `somalia_electricity_data_real.csv`
- `electricity_rf_model_real.pkl`
- `streamlit_app_real.py`

---

## 🚀 Deploy Online

You can deploy this project using [Streamlit Cloud](https://share.streamlit.io):

1. Push all files to a GitHub repo.
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and choose:
   - `streamlit_app_real.py` as the app file
4. Click **Deploy** and get a public link.

---

## 📁 Files in This Repository

- `README.md` — this file
- `somalia_electricity_data_real.csv` — dataset
- `somalia_electricity_model_real.py` — training script
- `electricity_rf_model_real.pkl` — trained model
- `streamlit_app_real.py` — dashboard script
- `Somalia_Electricity_Prediction_Report.pdf` — project summary (optional)

---

## 📘 License

MIT License — you are free to use, modify, and share this project with attribution.
