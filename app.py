import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="F1 Win Predictor", layout="centered")

st.title("🏎️ F1 Race Win Predictor")

st.markdown("Enter race details to predict win probability")

# Inputs
grid = st.slider("Grid Position", 1, 20, 5)
quali = st.slider("Qualifying Position", 1, 20, 5)
driver_age = st.slider("Driver Age", 18, 45, 28)

driver_prev_wins = st.number_input("Previous Wins", 0, 200, 5)
driver_prev_points = st.number_input("Previous Points", 0, 5000, 100)

constructor_prev_wins = st.number_input("Constructor Wins", 0, 500, 10)

# Dummy categorical inputs (match your training)
driverId = st.selectbox("Driver ID", ["1", "2", "3"])
constructorId = st.selectbox("Constructor ID", ["1", "2", "3"])
circuitId = st.selectbox("Circuit ID", ["1", "2", "3"])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "year": 2024,
        "round": 1,
        "grid": grid,
        "quali_position": quali,
        "driver_age": driver_age,
        "driver_prior_races": 50,
        "driver_prev_wins": driver_prev_wins,
        "driver_prev_points": driver_prev_points,
        "driver_prev_avg_finish": 5,
        "constructor_prior_races": 100,
        "constructor_prev_wins": constructor_prev_wins,
        "constructor_prev_points": 2000,
        "constructor_prev_avg_finish": 4,
        "driverId": driverId,
        "constructorId": constructorId,
        "circuitId": circuitId
    }])

    prob = model.predict_proba(input_data)[0][1]

    st.success(f"🏆 Win Probability: {prob*100:.2f}%")