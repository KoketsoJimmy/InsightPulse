# InsightPulse Insights Report
*Turn feedback into actionable insights.*

## Dataset summary
sample_feedback.csv (SYNTHETIC DATA): 520 records after cleaning. Labels: Provided 'sentiment' column.

## Sentiment distribution
- Positive: 273 (52.5%)
- Negative: 166 (31.9%)
- Neutral: 81 (15.6%)

## Model performance (held-out test set, 130 records)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 100.0% | 100.0% | 100.0% | 100.0% |
| Logistic Regression | 100.0% | 100.0% | 100.0% | 100.0% |

Best model: **Naive Bayes** (highest weighted F1).

## Key findings
- Most frequent words in negative feedback: slow (29), application (21), keeps (21), crashing (21), parcel (19).
- Dominant negative emotion: Frustration.
- Negative share moved from 28.1% (first half) to 35.8% (second half): worsening.
- Technology shows the largest rise in negative share (+26.1 points).
- Average sentiment score is 0.21 (scale -1 to +1).
- Technology has the highest negative share (42.5%, 45 comments).
- Pricing has the highest positive share (59.4%).

## AI-assisted insights
Overall sentiment is positive: 52.5% of 520 feedback records are positive (31.9% negative).

## Recommendations
- Investigate recurring application or system problems. (Technology: 42.5% negative, 106 records)
- Review delivery timelines and identify common causes of delays. (Delivery: 36.2% negative, 105 records)
- Monitor sentiment trends regularly and review a sample of predictions manually.

## Limitations
- Models can be wrong; sarcasm and context are hard.
- Emotion and category use simple keyword rules.
- Synthetic data does not represent real users; scores on it are optimistic.

## Conclusion
Use this report to support, not replace, human judgement.
