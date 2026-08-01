import io
import logging

import pandas as pd
import streamlit as st

from feature_sentiment_analysis import (
    VALID_FEATURES,
    load_csv_from_cos,
    analyze_satisfaction,
    build_model,
    predict_features,
    score_predictions,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Review Feature & Sentiment Analysis", layout="wide")
st.title(" Customer Review Feature & Sentiment Analysis")
st.caption(
    "Loads customer reviews, charts satisfaction by feature, and uses a "
    "watsonx.ai foundation model to predict which feature each review discusses."
)

for key in ["train_data", "test_data", "predictions_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("1. watsonx.ai credentials")
    wml_api_key = st.text_input("WML API key", type="password")
    wml_url = st.text_input("WML URL", value="https://us-south.ml.cloud.ibm.com")
    project_id = st.text_input("Project ID")
    model_id = st.text_input("Model ID", value="google/flan-t5-xxl")

    st.divider()
    st.header("2. Data source")
    source = st.radio("Load train/test data from", ["Upload CSV files", "IBM COS bucket"])

    cos_creds = None
    uploaded_train, uploaded_test = None, None

    if source == "IBM COS bucket":
        cos_api_key = st.text_input("COS API key", type="password")
        cos_bucket = st.text_input("COS bucket name")
        cos_endpoint = st.text_input(
            "COS endpoint URL",
            value="https://s3.private.us-south.cloud-object-storage.appdomain.cloud",
        )
        cos_auth_endpoint = st.text_input(
            "COS auth endpoint", value="https://iam.cloud.ibm.com/oidc/token"
        )
        if cos_api_key and cos_bucket:
            cos_creds = {
                "api_key": cos_api_key,
                "bucket": cos_bucket,
                "endpoint_url": cos_endpoint,
                "auth_endpoint": cos_auth_endpoint,
            }
    else:
        uploaded_train = st.file_uploader("train.csv", type="csv")
        uploaded_test = st.file_uploader("test.csv", type="csv")

    st.divider()
    load_clicked = st.button("📥 Load data", use_container_width=True)


if load_clicked:
    try:
        with st.spinner("Loading data..."):
            if source == "IBM COS bucket":
                if not cos_creds:
                    st.error("Enter your COS API key and bucket name first.")
                    st.stop()
                st.session_state.train_data = load_csv_from_cos(cos_creds, "train.csv")
                st.session_state.test_data = load_csv_from_cos(cos_creds, "test.csv")
            else:
                if not uploaded_train or not uploaded_test:
                    st.error("Upload both train.csv and test.csv first.")
                    st.stop()
                st.session_state.train_data = pd.read_csv(io.BytesIO(uploaded_train.getvalue()))
                st.session_state.test_data = pd.read_csv(io.BytesIO(uploaded_test.getvalue()))
        st.session_state.predictions_df = None  # reset downstream results
        st.success("Data loaded.")
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")

train_data = st.session_state.train_data
test_data = st.session_state.test_data

if train_data is None or test_data is None:
    st.info("Load train.csv and test.csv from the sidebar to get started.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Train data preview")
    st.dataframe(train_data.head(5), use_container_width=True)
with col2:
    st.subheader("Test data preview")
    st.dataframe(test_data.head(5), use_container_width=True)


st.header(" Satisfaction by feature")

if "Sentiment" not in test_data.columns or "Feature" not in test_data.columns:
    st.warning("test.csv needs 'Sentiment' and 'Feature' columns for this chart.")
else:
    result = (
        test_data[test_data["Sentiment"] == "Positive"]
        .groupby("Feature")["Sentiment"]
        .count()
        .sort_values(ascending=False)
    )

    if result.empty:
        st.warning("No positive-sentiment rows found.")
    else:
        st.bar_chart(result)
        m1, m2 = st.columns(2)
        # Ranked by count via idxmax/idxmin — not alphabetically by name.
        m1.metric(" Highly appreciated", result.idxmax(), f"{result.max()} positive reviews")
        m2.metric(" Least appreciated", result.idxmin(), f"{result.min()} positive reviews")


st.header("🤖 Predict feature from review text")

if "Review" not in train_data.columns:
    st.warning("train.csv needs a 'Review' column to run predictions.")
else:
    max_rows = len(train_data)
    sample_size = st.slider(
        "Number of reviews to run through the model",
        min_value=1,
        max_value=max_rows,
        value=min(20, max_rows),
        help="Each row is one call to watsonx.ai — start small to control cost/time.",
    )

    run_clicked = st.button(" Run prediction", type="primary")

    if run_clicked:
        if not wml_api_key or not project_id:
            st.error("Enter your WML API key and Project ID in the sidebar first.")
            st.stop()

        credentials = {"url": wml_url, "apikey": wml_api_key}
        subset = train_data.head(sample_size).copy()

        progress_bar = st.progress(0.0, text="Starting...")

        def _on_progress(done, total):
            progress_bar.progress(done / total, text=f"Processed {done}/{total} reviews")

        try:
            with st.spinner("Calling watsonx.ai..."):
                model = build_model(credentials, project_id, model_id=model_id)
                predictions = predict_features(
                    model, subset["Review"], progress_callback=_on_progress
                )
            subset["Predicted_Feature"] = predictions
            st.session_state.predictions_df = subset
            progress_bar.progress(1.0, text="Done")
            st.success("Prediction complete.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    predictions_df = st.session_state.predictions_df
    if predictions_df is not None:
        if "Feature" in predictions_df.columns:
            accuracy = (
                predictions_df["Feature"].str.strip().str.lower()
                == predictions_df["Predicted_Feature"]
            ).mean()
            st.metric("Accuracy", f"{accuracy * 100:.1f}%")

        st.dataframe(predictions_df, use_container_width=True)

        csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            " Download predictions CSV",
            data=csv_bytes,
            file_name="train_predictions.csv",
            mime="text/csv",
        )
