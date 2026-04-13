import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("model.pkl")

st.set_page_config(page_title="F1 Race Simulator", layout="wide")
st.title("🏎️ F1 Race Win Probability Simulator")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    base = Path("archive (2)")
    results = pd.read_csv(base / "results.csv", na_values=["\\N"])
    races = pd.read_csv(base / "races.csv", na_values=["\\N"])
    qualifying = pd.read_csv(base / "qualifying.csv", na_values=["\\N"])
    drivers = pd.read_csv(base / "drivers.csv", na_values=["\\N"])
    circuits = pd.read_csv(base / "circuits.csv", na_values=["\\N"])
    return results, races, qualifying, drivers, circuits


# -------------------------
# LOAD CIRCUITS
# -------------------------
@st.cache_data
def load_circuits():
    _, _, _, _, circuits = load_data()

    circuits["display_name"] = (
        circuits["name"] + " (" + circuits["location"] + ", " + circuits["country"] + ")"
    )

    circuit_map = dict(zip(circuits["display_name"], circuits["circuitId"].astype(str)))

    return circuit_map


# -------------------------
# BUILD DRIVER STATS
# -------------------------
@st.cache_data
def build_driver_stats():
    results, races, qualifying, drivers, _ = load_data()

    df = results.merge(races, on="raceId")
    df = df.merge(
        qualifying[["raceId", "driverId", "position"]],
        on=["raceId", "driverId"],
        how="left"
    )
    df = df.merge(drivers, on="driverId")

    df.rename(columns={"position": "quali_position"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    df = df.sort_values("date")

    df["win"] = (df["positionOrder"] == 1).astype(int)

    driver_group = df.groupby("driverId")
    constructor_group = df.groupby("constructorId")

    df["driver_prior_races"] = driver_group.cumcount()
    df["driver_prev_wins"] = driver_group["win"].cumsum() - df["win"]
    df["driver_prev_points"] = driver_group["points"].cumsum() - df["points"]

    df["driver_prev_avg_finish"] = (
        driver_group["positionOrder"].cumsum() - df["positionOrder"]
    ) / df["driver_prior_races"].replace(0, np.nan)

    df["constructor_prior_races"] = constructor_group.cumcount()
    df["constructor_prev_wins"] = constructor_group["win"].cumsum() - df["win"]
    df["constructor_prev_points"] = constructor_group["points"].cumsum() - df["points"]

    df["constructor_prev_avg_finish"] = (
        constructor_group["positionOrder"].cumsum() - df["positionOrder"]
    ) / df["constructor_prior_races"].replace(0, np.nan)

    df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25

    latest = df.sort_values("date").groupby("driverId").tail(1)
    latest = latest.fillna(0)

    return latest


# -------------------------
# DRIVER SELECTION
# -------------------------
stats_df = build_driver_stats()

driver_names = stats_df["forename"] + " " + stats_df["surname"]
driver_map = dict(zip(driver_names, stats_df["driverId"]))

selected_drivers = st.multiselect(
    "Select Drivers",
    options=list(driver_map.keys()),
    default=list(driver_map.keys())[:10]
)

# -------------------------
# GRID ASSIGNMENT (UNIQUE)
# -------------------------
st.subheader("Grid Assignment")

grid_inputs = {}
quali_inputs = {}

available_positions = list(range(1, 21))

for driver in selected_drivers:
    remaining_positions = [p for p in available_positions if p not in grid_inputs.values()]

    col1, col2 = st.columns(2)

    with col1:
        grid_inputs[driver] = st.selectbox(
            f"{driver} Grid",
            options=remaining_positions,
            key=f"grid_{driver}"
        )

    with col2:
        quali_inputs[driver] = st.number_input(
            f"{driver} Quali",
            min_value=1,
            max_value=20,
            value=grid_inputs[driver],
            key=f"quali_{driver}"
        )

# -------------------------
# CIRCUIT SELECTION
# -------------------------
st.subheader("🏁 Select Circuit")

circuit_map = load_circuits()

selected_circuit = st.selectbox(
    "Choose Track",
    options=list(circuit_map.keys())
)

circuit_id = circuit_map[selected_circuit]

# -------------------------
# PREDICTION
# -------------------------
if st.button("Simulate Race 🏁"):

    rows = []

    for driver in selected_drivers:
        d_id = driver_map[driver]
        row = stats_df[stats_df["driverId"] == d_id].iloc[0]

        rows.append({
            "year": 2024,
            "round": 1,
            "grid": grid_inputs[driver],
            "quali_position": quali_inputs[driver],
            "driver_age": row["driver_age"],
            "driver_prior_races": row["driver_prior_races"],
            "driver_prev_wins": row["driver_prev_wins"],
            "driver_prev_points": row["driver_prev_points"],
            "driver_prev_avg_finish": row["driver_prev_avg_finish"],
            "constructor_prior_races": row["constructor_prior_races"],
            "constructor_prev_wins": row["constructor_prev_wins"],
            "constructor_prev_points": row["constructor_prev_points"],
            "constructor_prev_avg_finish": row["constructor_prev_avg_finish"],
            "driverId": str(int(row["driverId"])),
            "constructorId": str(int(row["constructorId"])),
            "circuitId": str(circuit_id)
        })

    input_df = pd.DataFrame(rows)

    # Raw probabilities
    probs = model.predict_proba(input_df)[:, 1]

    # Normalize for UI
    probs = probs / probs.sum()

    input_df["Driver"] = selected_drivers
    input_df["Win Probability (%)"] = (probs * 100).round(2)

    result = input_df.sort_values("Win Probability (%)", ascending=False).reset_index(drop=True)

    st.subheader("🏆 Race Prediction Leaderboard")

    # Table
    st.dataframe(
        result[["Driver", "grid", "Win Probability (%)"]],
        use_container_width=True
    )

    # Leaderboard
    st.subheader("📊 Podium View")

    for i, row in result.iterrows():
        st.metric(
            label=f"#{i+1} {row['Driver']}",
            value=f"{row['Win Probability (%)']}%"
        )

    winner = result.iloc[0]

    st.success(
        f"🏁 Predicted Winner: {winner['Driver']} ({winner['Win Probability (%)']}%)"
    )