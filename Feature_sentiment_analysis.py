import os
import sys
import time
import getpass
import types
import logging
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless/script execution
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_FEATURES = ["camera quality", "display quality", "battery life", "performance speed"]

FEW_SHOT_INSTRUCTION = """Identify the product feature being discussed in the following customer review. Choose only one of the following features: camera quality, display quality, battery life, or performance speed.

review: The pictures are blurry and lack detail.
feature: camera quality

review: I need to charge the phone multiple times a day.
feature: battery life

review: Apps open instantly and the phone feels smooth.
feature: performance speed

review: The display quality is top-notch and very vibrant.
feature: display quality

review: {review}
feature:"""



def get_wml_credentials() -> dict:
    url = os.environ.get("WML_URL", "https://us-south.ml.cloud.ibm.com")
    apikey = os.environ.get("WML_API_KEY") or getpass.getpass("Enter your WML API key: ")
    return {"url": url, "apikey": apikey}


def get_project_id() -> str:
    project_id = os.environ.get("PROJECT_ID")
    if not project_id:
        project_id = input("Enter your watsonx project_id: ").strip()
    return project_id


def get_cos_credentials() -> dict:
    """Load COS credentials from env vars, prompting if missing. Never hardcode these."""
    api_key = os.environ.get("COS_API_KEY") or getpass.getpass("Enter your COS API key: ")
    endpoint_url = os.environ.get(
        "COS_ENDPOINT_URL", "https://s3.private.us-south.cloud-object-storage.appdomain.cloud"
    )
    auth_endpoint = os.environ.get("COS_AUTH_ENDPOINT", "https://iam.cloud.ibm.com/oidc/token")
    bucket = os.environ.get("COS_BUCKET") or input("Enter your COS bucket name: ").strip()
    return {
        "api_key": api_key,
        "endpoint_url": endpoint_url,
        "auth_endpoint": auth_endpoint,
        "bucket": bucket,
    }


def load_csv_from_cos(cos_creds: dict, object_key: str) -> pd.DataFrame:
    
    import ibm_boto3
    from botocore.client import Config

    cos_client = ibm_boto3.client(
        service_name="s3",
        ibm_api_key_id=cos_creds["api_key"],
        ibm_auth_endpoint=cos_creds["auth_endpoint"],
        config=Config(signature_version="oauth"),
        endpoint_url=cos_creds["endpoint_url"],
    )

    body = cos_client.get_object(Bucket=cos_creds["bucket"], Key=object_key)["Body"]

    if not hasattr(body, "__iter__"):
        body.__iter__ = types.MethodType(lambda self: iter(self.read().splitlines(True)), body)

    df = pd.read_csv(body)
    log.info("Loaded %s (%d rows) from bucket '%s'", object_key, len(df), cos_creds["bucket"])
    return df


def load_local_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %s (%d rows) from local disk", path, len(df))
    return df


def load_dataset(name: str, cos_creds: Optional[dict]) -> pd.DataFrame:

    env_var = "TRAIN_CSV_PATH" if name == "train" else "TEST_CSV_PATH"
    local_path = os.environ.get(env_var)
    if local_path:
        return load_local_csv(local_path)
    if cos_creds is None:
        raise RuntimeError(f"No local path for {name}.csv and no COS credentials provided.")
    return load_csv_from_cos(cos_creds, f"{name}.csv")



def analyze_satisfaction(test_data: pd.DataFrame) -> pd.Series:
    """Count positive-sentiment rows per feature and chart them."""
    result = (
        test_data[test_data["Sentiment"] == "Positive"]
        .groupby("Feature")["Sentiment"]
        .count()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(result.index, result.values)
    plt.title("Customer Satisfaction by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Positive review count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "satisfaction_by_feature.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    log.info("Saved chart to %s", chart_path)

    if not result.empty:
        # Fixed: rank by COUNT (idxmax/idxmin), not alphabetically by feature name.
        log.info("HIGHLY APPRECIATED FEATURE: %s (%d positive reviews)",
                  result.idxmax(), result.max())
        log.info("LEAST APPRECIATED FEATURE: %s (%d positive reviews)",
                  result.idxmin(), result.min())

    return result


def build_model(credentials: dict, project_id: str, model_id: Optional[str] = None):
    from ibm_watson_machine_learning.foundation_models import Model
    from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

    # google/flan-ul2 is deprecated; default to an actively supported model,
    # but allow override via env var / argument.
    model_id = model_id or os.environ.get("WML_MODEL_ID", "google/flan-t5-xxl")

    parameters = {
        GenParams.MAX_NEW_TOKENS: 10,
        GenParams.DECODING_METHOD: "greedy",
    }

    return Model(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)


def parse_predicted_feature(raw_text: str) -> str:
    """Normalize model output to one of VALID_FEATURES, or 'unknown'."""
    text = raw_text.strip().lower()
    for feature in VALID_FEATURES:
        if feature in text:
            return feature
    return "unknown"


def predict_features(
    model,
    reviews: pd.Series,
    sleep_seconds: float = 0.6,
    max_retries: int = 3,
    progress_callback=None,
) -> list:
    """
    progress_callback, if given, is called as progress_callback(done, total)
    after each review — used by the Streamlit UI to drive a progress bar.
    """
    predictions = []
    total = len(reviews)
    for i, review_text in enumerate(reviews, start=1):
        prompt = FEW_SHOT_INSTRUCTION.format(review=review_text)

        raw_output = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_output = model.generate_text(prompt=prompt)
                break
            except Exception as exc:  # network/rate-limit errors from the API
                log.warning("Generation failed (attempt %d/%d) for row %d: %s",
                            attempt, max_retries, i, exc)
                time.sleep(sleep_seconds * attempt)

        predicted = parse_predicted_feature(raw_output) if raw_output else "unknown"
        predictions.append(predicted)

        if i % 10 == 0 or i == total:
            log.info("Processed %d/%d reviews", i, total)
        if progress_callback:
            progress_callback(i, total)

        time.sleep(sleep_seconds)  # stay under rate limits

    return predictions


def score_predictions(actual: pd.Series, predicted: list) -> float:
    correct = sum(a.strip().lower() == p for a, p in zip(actual, predicted))
    accuracy = correct / len(predicted) if predicted else 0.0
    log.info("Accuracy: %d/%d (%.1f%%)", correct, len(predicted), accuracy * 100)
    return accuracy


def main():
    use_cos = os.environ.get("USE_COS", "1") != "0"

    cos_creds = get_cos_credentials() if use_cos else None
    train_data = load_dataset("train", cos_creds)
    test_data = load_dataset("test", cos_creds)

    log.info("Train sample:\n%s", train_data.head(5).to_string())
    log.info("Test sample:\n%s", test_data.head(5).to_string())

    analyze_satisfaction(test_data)

    wml_credentials = get_wml_credentials()
    project_id = get_project_id()
    model = build_model(wml_credentials, project_id)

    predictions = predict_features(model, train_data["Review"])
    train_data = train_data.copy()
    train_data["Predicted_Feature"] = predictions

    score_predictions(train_data["Feature"], predictions)

    results_path = os.path.join(OUTPUT_DIR, "train_predictions.csv")
    train_data.to_csv(results_path, index=False)
    log.info("Saved predictions to %s", results_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user.")
