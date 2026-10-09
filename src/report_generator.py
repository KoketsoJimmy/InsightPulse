"""Downloadable report (Markdown, plus HTML if the 'markdown' package is installed)."""
import pandas as pd
from src import insights as ins

def _table(df: pd.DataFrame) -> str:
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))

def build_report(R: dict, df: pd.DataFrame, name: str) -> str:
    n, q, vc = len(df), R["quality"], df.sentiment.value_counts()
    t = ins.trend(df); ti = ins.top_issues(df)
    perf = pd.DataFrame(R["metrics"]).T.reset_index().rename(columns={"index": "Model"})
    L = lambda xs: "\n".join(f"- {x}" for x in xs)
    return f"""# InsightPulse Report
*Customer Feedback Intelligence*

## 1. Dataset overview
{name}: {n} records analysed ({q['rows']} raw rows, {q['columns']} columns). Labels: **{R['label_source']}**.

## 2. Data quality
Data Quality Score: **{q['score']}/100**
{L(m for _, m in q['checks'])}

## 3. Sentiment summary
{L(f"{k}: {v} ({v / n * 100:.1f}%)" for k, v in vc.items())}

## 4. Sentiment trends
{f"Negative share moved from {t[0]:.1f}% to {t[1]:.1f}% between the first and second half of the period ({t[2]})." if t else "No usable dates, so no trend is reported."}

## 5. Top issues
{_table(ti) if len(ti) else "No negative feedback found."}

## 6. Model performance
Best model: **{R['best']}** (highest weighted F1 on {R['ntest']} test records).

{ins.explain_model(R)}

## 7. Model comparison
{_table(perf)}

## 8. Executive summary
{L(f"**{h}:** {x}" for h, x in ins.exec_summary(df))}

## Recommendations
{L(f"**{r['priority']} - {r['topic']}:** {r['evidence']} {r['action']}" for r in ins.prioritised_recs(df))}

## Important data limitations
- Models can be wrong; sarcasm and context are hard to detect.
- Topics and emotions use simple keyword rules.
- Pseudo-labels and synthetic data do not show real-world performance.
- This is a demonstration tool; validate before using it for decisions.
"""

def to_html(md: str) -> str:
    try: import markdown; body = markdown.markdown(md, extensions=["tables"])
    except ImportError: body = "<pre>" + md.replace("<", "&lt;") + "</pre>"
    return f"<!doctype html><meta charset=utf-8><title>InsightPulse Report</title><body style=\"font-family:system-ui;max-width:820px;margin:2rem auto;padding:0 1rem;line-height:1.5\"><style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px}}</style>{body}</body>"
