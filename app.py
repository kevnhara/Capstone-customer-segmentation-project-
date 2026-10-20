
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Locate the saved model and scaler
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "customer_segmentation_artifacts"

MODEL_PATH = ARTIFACTS_DIR / "kmeans_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"

st.set_page_config(page_title="Customer Segmentation",
                   page_icon="📊")

st.title("Customer Segmentation App")
st.write(
    "Enter a customer's annual income and spending score "
    "to identify their customer segment."
)

# Check that the saved files exist
if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error(
        "Model or scaler not found. Check the "
        "customer_segmentation_artifacts folder."
    )
    st.stop()

# Load the trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Collect customer information
income = st.number_input(
    "Annual income (in thousands)",
    min_value=0.0,
    value=50.0,
    step=1.0
)

spending_score = st.slider(
    "Spending score",
    min_value=1,
    max_value=100,
    value=50
)

# Predict the customer's cluster
if st.button("Identify Customer Segment"):
    customer = pd.DataFrame(
        [[income, spending_score]],
        columns=[
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    
    try:
        scaled_customer = scaler.transform(customer)
        cluster = model.predict(scaled_customer)[0]

        # Give each cluster a meaningful label
        cluster_labels = {
            0: "Average Income, Average Spending",
            1: "High Income, High Spending",
            2: "Low Income, High Spending",
            3: "High Income, Low Spending",
            4: "Low Income, Low Spending"
        }

        descriptions = {
            0: "Customers with moderate income and moderate spending.",
            1: "Customers with high income and high spending scores.",
            2: "Customers with low income and high spending scores.",
            3: "Customers with high income and low spending scores.",
            4: "Customers with low income and low spending scores."
        }

        label = cluster_labels.get(
            int(cluster),
            "Unidentified Customer Segment"
        )

        st.success(f"Customer Segment: {label}")
        st.write(f"Assigned cluster number: {cluster}")
        st.write(
            descriptions.get(
                int(cluster),
                "No description available."
            )
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the feature names and order match "
            "those used when training your model and scaler."
        )
