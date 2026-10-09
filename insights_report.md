# InsightPulse Report
*Customer Feedback Intelligence*

## 1. Dataset overview
demo (SYNTHETIC): 240 records analysed (240 raw rows, 5 columns). Labels: **Provided labels ('sentiment')**.

## 2. Data quality
Data Quality Score: **100/100**
- No duplicate records
- No missing values
- No empty feedback records
- 240 rows
- Average feedback length: 7.7 words
- All sentiment labels valid

## 3. Sentiment summary
- Positive: 142 (59.2%)
- Negative: 63 (26.2%)
- Neutral: 35 (14.6%)

## 4. Sentiment trends
Negative share moved from 24.2% to 28.3% between the first and second half of the period (worsening).

## 5. Top issues
| Topic | Complaints | Share % |
|---|---|---|
| Website/App | 14 | 22.2 |
| Customer Service | 11 | 17.5 |
| Pricing | 10 | 15.9 |
| Returns/Refunds | 10 | 15.9 |
| Delivery | 9 | 14.3 |
| Product Quality | 9 | 14.3 |

## 6. Model performance
Best model: **Logistic Regression** (highest weighted F1 on 60 test records).

On 60 unseen test records, Logistic Regression classified about 100% correctly (accuracy). Of the records it labelled as a class, about 100% were right (precision); of the records truly in a class, it found about 100% (recall). F1 (100%) balances the two. These figures are measured against the labels provided in the dataset, and depend on dataset size and quality.

## 7. Model comparison
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 98.3 | 98.4 | 98.3 | 98.3 |
| Logistic Regression | 100.0 | 100.0 | 100.0 | 100.0 |
| Random Forest | 100.0 | 100.0 | 100.0 | 100.0 |
| Linear SVM | 100.0 | 100.0 | 100.0 | 100.0 |

## 8. Executive summary
- **Overall sentiment:** Overall sentiment is predominantly positive (59.2% of 240 records).
- **Most common issue:** Website/App (14 complaints, 22.2% of negative feedback).
- **Important trend:** Negative feedback moved from 24.2% to 28.3% between the first and second half of the period (worsening).
- **Biggest concern:** Delivery has the highest negative share (37.5%).
- **Positive finding:** Other has the highest positive share (100.0%).
- **Recommended focus:** Investigate recurring website or app problems and test the checkout flow.

## Recommendations
- **HIGH - Delivery:** 37.5% negative (9 of 24 comments) versus 26.2% overall. Review delivery timelines and identify common causes of delays.
- **MEDIUM - Product Quality:** 31.0% negative (9 of 29 comments) versus 26.2% overall. Inspect product quality control and review defect reports.
- **MEDIUM - Website/App:** 26.4% negative (14 of 53 comments) versus 26.2% overall. Investigate recurring website or app problems and test the checkout flow.
- **MEDIUM - Returns/Refunds:** 26.3% negative (10 of 38 comments) versus 26.2% overall. Review refund processing times and the returns procedure.
- **MEDIUM - Pricing:** 25.6% negative (10 of 39 comments) versus 26.2% overall. Review pricing and fee transparency.
- **MEDIUM - Customer Service:** 23.9% negative (11 of 46 comments) versus 26.2% overall. Consider investigating customer-service response and waiting times.
- **LOW - Other:** 0.0% negative (0 of 11 comments) versus 26.2% overall. Continue monitoring other feedback.

## Important data limitations
- Models can be wrong; sarcasm and context are hard to detect.
- Topics and emotions use simple keyword rules.
- Pseudo-labels and synthetic data do not show real-world performance.
- This is a demonstration tool; validate before using it for decisions.
