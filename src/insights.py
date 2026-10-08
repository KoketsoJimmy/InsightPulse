"""Local, data-driven insights, recommendations, Q&A and report."""
import re, pandas as pd
from src.data_processing import tokens
SC = {"Positive": 1, "Neutral": 0, "Negative": -1}
def filt(df, a):
    for k in ("sentiment", "emotion", "topic"):
        if a.get(k): df = df[df[k] == a[k]]
    if a.get("start"): df = df[df["date"] >= a["start"]]
    if a.get("end"): df = df[df["date"] <= a["end"]]
    if a.get("q"): df = df[df["text"].str.contains(re.escape(a["q"]), case=False)]
    return df

SC = {"Positive": 1, "Neutral": 0, "Negative": -1}
def insights(df):
    n = len(df); out = {"summary": "", "findings": [], "positives": [], "negatives": [], "recs": []}
    vc = df.sentiment.value_counts(); top = vc.idxmax()
    out["summary"] = f"Overall sentiment is {top.lower()}: {vc[top]/n*100:.1f}% of {n} feedback records are {top.lower()} ({(df.sentiment=='Negative').mean()*100:.1f}% negative)."
    g = df.groupby("topic").agg(n=("sentiment", "size"), neg=("sentiment", lambda s: (s == "Negative").mean()), pos=("sentiment", lambda s: (s == "Positive").mean()), negc=("sentiment", lambda s: (s == "Negative").sum()))
    g = g[g.n >= 5]
    if len(g):
        wn, bp = g.neg.idxmax(), g.pos.idxmax()
        out["negatives"].append(f"{wn} has the highest negative share ({g.neg[wn]*100:.1f}%, {int(g.negc[wn])} comments).")
        out["positives"].append(f"{bp} has the highest positive share ({g.pos[bp]*100:.1f}%).")
        out["recs"].append(f"Investigate recurring {wn.lower()} complaints first; they affect the most customers.")
    neg = df[df.sentiment == "Negative"]
    if len(neg):
        kw = pd.Series([w for t in neg.text for w in tokens(t)]).value_counts().head(5)
        out["findings"].append("Most frequent words in negative feedback: " + ", ".join(f"{k} ({v})" for k, v in kw.items()) + ".")
        out["recs"].append(f"Review comments mentioning '{kw.index[0]}', the most common negative keyword.")
        out["findings"].append(f"Dominant negative emotion: {neg.emotion.value_counts().idxmax()}.")
    d = df.dropna(subset=["date"])
    if d.date.nunique() > 1:
        mid = d.date.min() + (d.date.max() - d.date.min()) / 2; a, b = d[d.date <= mid], d[d.date > mid]
        if len(a) and len(b):
            x, y = (a.sentiment == "Negative").mean() * 100, (b.sentiment == "Negative").mean() * 100
            out["findings"].append(f"Negative share moved from {x:.1f}% (first half) to {y:.1f}% (second half): {'worsening' if y > x + 1 else 'improving' if y < x - 1 else 'stable'}.")
            tg = d.groupby("topic").apply(lambda t: (t[t.date > mid].sentiment == "Negative").mean() - (t[t.date <= mid].sentiment == "Negative").mean() if len(t[t.date > mid]) and len(t[t.date <= mid]) else 0)
            if len(tg) and tg.max() > .05: out["findings"].append(f"{tg.idxmax()} shows the largest rise in negative share (+{tg.max()*100:.1f} points).")
    out["recs"].append("Monitor sentiment trends weekly and re-run this analysis on new feedback.")
    out["findings"].append(f"Average sentiment score is {df.sentiment.map(SC).mean():.2f} (scale -1 to +1).")
    return out

def ask(df, q):
    q = q.lower(); n = len(df)
    nc = lambda s: int((df.sentiment == s).sum())
    for s in ("positive", "negative", "neutral"):
        if "how many" in q and s in q: return f"{nc(s.title())} of {n} comments ({nc(s.title())/n*100:.1f}%) are {s}."
    if "emotion" in q: vc = df.emotion.value_counts(); return f"The most common emotion is {vc.idxmax()} ({vc.iloc[0]} comments, {vc.iloc[0]/n*100:.1f}%)."
    if any(k in q for k in ("improv", "over time", "trend")):
        i = insights(df); return next((f for f in i["findings"] if "moved" in f), "Not enough dated data to assess a trend.")
    if "highest positive" in q or "most positive" in q or "best" in q:
        g = (df.assign(p=df.sentiment == "Positive").groupby("topic").p.mean()); return f"{g.idxmax()} has the highest positive share ({g.max()*100:.1f}%)."
    if any(k in q for k in ("worst", "most negative", "causing", "negative sentiment", "highest negative")):
        g = df.assign(p=df.sentiment == "Negative").groupby("topic").p.agg(["mean", "sum"]); t = g["mean"].idxmax()
        neg = df[(df.sentiment == "Negative") & (df.topic == t)]; kw = pd.Series([w for x in neg.text for w in tokens(x)]).value_counts().head(3)
        return f"{t} is the worst: {g['mean'][t]*100:.1f}% negative ({int(g['sum'][t])} comments). Top words: {', '.join(kw.index)}."
    if "topic" in q or "category" in q: vc = df.topic.value_counts(); return f"The most common topic is {vc.idxmax()} ({vc.iloc[0]} comments)."
    return "I can answer questions about sentiment counts, emotions, topics, worst/best topic and trends over time. Try: 'Which topic has the worst sentiment?'"


RECS = {"Customer Service": "Consider investigating customer-service response and waiting times.",
        "Technology": "Investigate recurring application or system problems.",
        "Delivery": "Review delivery timelines and identify common causes of delays.",
        "Pricing": "Review pricing and fee transparency.", "Quality": "Inspect product quality control and returns.",
        "Support": "Review support response and resolution times.", "Staff": "Consider staff training and feedback sessions.",
        "Product": "Review product features against customer expectations."}
def recommendations(df, thr=0.30):
    g = df.groupby("topic").agg(n=("sentiment", "size"), neg=("sentiment", lambda s: (s == "Negative").mean())); g = g[g.n >= 5].sort_values("neg", ascending=False)
    out = [f"{RECS.get(t, 'Investigate recurring complaints.')} ({t}: {r.neg*100:.1f}% negative, {int(r.n)} records)" for t, r in g.iterrows() if r.neg >= thr][:4]
    return out + ["Monitor sentiment trends regularly and review a sample of predictions manually."]
def report_md(R, df, name):
    i = insights(df); vc = df.sentiment.value_counts(); n = len(df)
    mt = "\n".join(f"| {k} | {v['Accuracy']}% | {v['Precision']}% | {v['Recall']}% | {v['F1']}% |" for k, v in R["metrics"].items())
    L = lambda xs: "\n".join(f"- {x}" for x in xs)
    return f"""# InsightPulse Insights Report
*Turn feedback into actionable insights.*

## Dataset summary
{name}: {n} records after cleaning. Labels: {R['label_source']}.

## Sentiment distribution
{L(f"{k}: {v} ({v/n*100:.1f}%)" for k, v in vc.items())}

## Model performance (held-out test set, {R['ntest']} records)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
{mt}

Best model: **{R['best']}** (highest weighted F1).

## Key findings
{L(i['findings'] + i['negatives'] + i['positives'])}

## AI-assisted insights
{i['summary']}

## Recommendations
{L(recommendations(df))}

## Limitations
- Models can be wrong; sarcasm and context are hard.
- Emotion and category use simple keyword rules.
- Synthetic data does not represent real users; scores on it are optimistic.

## Conclusion
Use this report to support, not replace, human judgement.
"""
