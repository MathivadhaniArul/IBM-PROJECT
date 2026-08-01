# Feature & Sentiment Analysis Pipeline

Classifies customer reviews by product feature — camera quality, display
quality, battery life, performance speed — and shows which features
customers are happiest and unhappiest with. Classification is done by an
IBM watsonx.ai foundation model.

## Features
- Loads review data (CSV upload, or from an IBM COS bucket)
- Bar chart of positive-review counts per feature, with the top and
  bottom feature called out
- Predicts the feature discussed in each review using a watsonx.ai model
  (few-shot prompting, no training needed)
- Compares predictions to the labeled data and reports accuracy
- Download results as CSV

## Setup
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it (Streamlit UI)
```bash
streamlit run app.py
```
1. Enter your watsonx API key, URL, and Project ID in the sidebar
2. Upload `train.csv` / `test.csv`, or pull them from a COS bucket
3. Click **Load data** to see the satisfaction chart
4. Pick a sample size and click **Run prediction** to classify reviews
5. Download the results as CSV

## Run it (command line)
Set these as environment variables, then run the script:

| Variable | Notes |
|---|---|
| `WML_API_KEY`, `PROJECT_ID` | watsonx credentials (prompted if unset) |
| `WML_MODEL_ID` | model to use, default `google/flan-t5-xxl` |
| `COS_API_KEY`, `COS_BUCKET` | only needed if loading from COS |
| `USE_COS=0` | skip COS, use local files instead |
| `TRAIN_CSV_PATH`, `TEST_CSV_PATH` | needed if `USE_COS=0` |

```bash
export WML_API_KEY="..." PROJECT_ID="..." COS_API_KEY="..." COS_BUCKET="..."
python feature_sentiment_analysis.py
```
