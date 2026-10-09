# InsightPulse: Customer Feedback Intelligence
Turn customer feedback into actionable insights using data analytics, machine learning and AI. Python + Streamlit, fully offline after installing dependencies. No APIs, databases or cloud services.

## Features
- Demo mode ("Load Demo Dataset", 240 synthetic records) and CSV upload with a 5-step workflow (upload, feedback column, sentiment column, validate, analyse)
- Data Quality score (0-100) with green/orange/red checks
- Sentiment analysis with confidence and score; topic detection (Delivery, Customer Service, Product Quality, Pricing, Returns/Refunds, Website/App, Payment, Other)
- Dashboard, Topics & Trends, Feedback Explorer (search, filters, sorting, record view)
- AI Executive Summary, Ask the Data, prioritised Recommendations (deterministic Python rules, every number computed from the data)
- Model Performance (accuracy, precision, recall, F1, confusion matrix) and Model Comparison (Naive Bayes, Logistic Regression, Random Forest, Linear SVM on one shared split)
- Downloadable report (HTML and Markdown)

## Stack
Python 3, Streamlit, pandas, NumPy, scikit-learn, NLTK, Plotly, Matplotlib, joblib, markdown.

## Structure
`app.py` (navigation) | `views.py` (pages) | `src/data_processing.py` | `src/sentiment_analysis.py` | `src/model_training.py` | `src/insights.py` | `src/report_generator.py` | `src/visualizations.py` | `tests/` | `data/` | `models/` | `reports/` | `documentation/`

## Install and run
```
pip install -r requirements.txt
streamlit run app.py
```

## Demo mode / presentation flow
Open the app, click **Load Demo Dataset**, then: Dashboard, Sentiment Analysis, Topics & Trends, Ask the Data, Model Performance, Recommendations, Generate Report > Download.

## Supported CSV structure
One text column (auto-detected: text, feedback, comment, review, message, response). Optional: `sentiment` (Positive/Neutral/Negative), `date`, `rating`, `category`. Without a sentiment column the app uses keyword **pseudo-labels** and says so everywhere.

## Machine learning
TF-IDF (1-2 grams) with four classifiers, 75/25 stratified split, weighted precision/recall/F1. The best F1 wins; ties go to the simpler model. Models that cannot train are reported, never faked.

## Testing
`python -m pytest -q tests` (unit tests, plus every page loaded through Streamlit's test runner).

## Known limitations
Single-user, in-memory state. Rule-based topics/emotions. Sarcasm and context are hard. The synthetic demo is template-generated, so its scores are optimistic. Metrics on pseudo-labelled data only measure agreement with the keyword rule. No native PDF export (print the HTML report to PDF).

## Data & AI notice
Synthetic/demo data may be used; labels may be pseudo-labels; performance depends on data quality and size; this is a demonstration tool and needs validation and human oversight before informing decisions.
