import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# ------------------ STREAMLIT APP ------------------
current_date = pd.Timestamp.now(tz="utc")
current_date = current_date.replace(year=2024)
st.title("NewYork City Citi Bike Trip Prediction")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

progress_bar = st.sidebar.progress(0)
N_STEPS = 3

# Step 1: Load features
with st.spinner("Loading batch of features from feature store..."):
    features,targets = load_batch_of_features_from_store(current_date)
    features['target'] = targets
    st.sidebar.write("Features loaded.")
    progress_bar.progress(1 / N_STEPS)

# Step 2: Fetch predictions
with st.spinner("Fetching predictions from latest model..."):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Predictions ready.")
    progress_bar.progress(2 / N_STEPS)

# Step 3: Prediction Stats
st.subheader("Prediction Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Trips", f"{predictions['predicted_demand'].mean():.0f}")
col2.metric("Max Trips", f"{predictions['predicted_demand'].max():.0f}")
col3.metric("Min Trips", f"{predictions['predicted_demand'].min():.0f}")
progress_bar.progress(3 / N_STEPS)

# ------------------ Top 10 Stations ------------------
st.subheader("Top 10 Stations by Predicted Demand")
top10 = predictions.sort_values("predicted_demand", ascending=False).head(10)
st.dataframe(top10[["pickup_location_id", "predicted_demand"]])

# ------------------ Dropdown + Plot ------------------
st.subheader("Prediction Trend for Selected Station")
selected_id = st.selectbox("Select Station ID", predictions["pickup_location_id"].unique())

filtered_features = features[features["pickup_location_id"] == selected_id]
filtered_predictions = predictions[predictions["pickup_location_id"] == selected_id]

if not filtered_features.empty and not filtered_predictions.empty:
    fig = plot_prediction(
        features=filtered_features,
        prediction=filtered_predictions,
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
else:
    st.warning("No data available for the selected station.")
