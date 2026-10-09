# InsightPulse Technical Documentation (condensed)
Expand each point below into a section before submission.
1. Data cleaning: `dp.load_clean` drops missing text and duplicates, keeps the original frame.
2. NLP preprocessing: `dp.clean` (lowercase, URL and symbol removal), `dp.tokens` (NLTK tokeniser, stop words, negations kept).
3. TF-IDF: weights words by frequency in a comment versus rarity across all comments.
4. Models: Naive Bayes and Logistic Regression in `sa.train`; best weighted F1 chosen.
5. Metrics: accuracy, precision, recall, F1 on a held-out 25%.
6. Emotion and category: transparent keyword lexicons in `sentiment_analysis.py`.
7. Insights: `insights.py`, every figure computed from the data.
8. Testing: `tests/test_core.py` (13 tests).
9. Ethics and limitations: see README.
