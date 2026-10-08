# InsightPulse: Sentiment Analysis & Data Insights Dashboard
*Turn feedback into actionable insights.* CAPACITI Week 3 individual project. Python-first, runs fully offline.

## Overview and problem
Organisations collect written feedback but cannot read thousands of comments. InsightPulse turns a feedback CSV into sentiment, emotion, category, trends, model metrics, insights and recommendations.

## Objectives
Interpret data with AI, build a real sentiment classifier, visualise results, generate data-driven insights, document the work.

## Features
Dataset upload with column selection, 6 Plotly charts, KPI cards, filters and keyword search, category analysis, single-feedback analyser, model comparison, local insights and recommendations, Ask the Data, downloadable Markdown report.

## Stack
Python 3, Streamlit, pandas, NumPy, scikit-learn, NLTK, Plotly, Matplotlib, joblib. No APIs, no cloud.

## Architecture
`app.py` (UI) -> `src/data_processing.py` (pandas + NLTK cleaning) -> `src/sentiment_analysis.py` (TF-IDF, models, emotion, category) -> `src/insights.py` (insights, recommendations, report) -> `src/visualizations.py` (charts).

## Install and run
```
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q tests      # run tests
```
Use: open **Dataset**, load the sample or upload a CSV, then explore the other pages. Needs internet only for `pip install`.

## Machine-learning method
Text is lowercased, URLs and special characters removed, tokenised and stop words removed with NLTK (negations like "not" are kept). TF-IDF (1-2 grams) feeds Naive Bayes and Logistic Regression. A stratified 75/25 split gives real accuracy, weighted precision, recall and F1; the best F1 wins. Confidence is the top class probability; score is P(positive) minus P(negative).
Without a `sentiment` column, keyword-lexicon pseudo-labels are used and flagged.

## Insights
All numbers are computed with pandas. Templates only supply wording; recommendations trigger when a category has 30% or more negative feedback.

## Results
Shown live on **Model Performance**. The synthetic sample is template-generated, so its scores are near-perfect and not representative of real data.

## Challenges / lessons
Keeping insights data-driven, avoiding label leakage, handling messy CSVs (add your own notes).

## Ethics and limitations
Synthetic data, local processing, possible bias, sarcasm and context errors, rule-based emotion and categories, human oversight required. Educational prototype.

## Screenshots
Add screenshots to `screenshots/`.

## Future improvements
Transformer models, multilingual support, scheduled reports, database, authentication, cloud deployment.

## Author
Your name, CAPACITI
