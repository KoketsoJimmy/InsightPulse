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

def trend(df):
    """Negative share in first vs second half of the date range, or None without dates."""
    d = df.dropna(subset=["date"])
    if d.date.nunique() < 2: return None
    mid = d.date.min() + (d.date.max() - d.date.min()) / 2; a, b = d[d.date <= mid], d[d.date > mid]
    if not len(a) or not len(b): return None
    x, y = (a.sentiment == "Negative").mean() * 100, (b.sentiment == "Negative").mean() * 100
    return x, y, ("worsening" if y > x + 1 else "improving" if y < x - 1 else "stable")

def top_issues(df):
    """Topics ranked by number of negative comments (complaints)."""
    neg = df[df.sentiment == "Negative"]
    if neg.empty: return pd.DataFrame(columns=["Topic", "Complaints", "Share %"])
    t = neg.topic.value_counts().rename_axis("Topic").reset_index(name="Complaints"); t["Share %"] = (t.Complaints / len(neg) * 100).round(1); return t

def topic_table(df):
    g = df.groupby("topic").agg(Feedback=("text", "size"), Positive=("sentiment", lambda s: (s == "Positive").sum()), Negative=("sentiment", lambda s: (s == "Negative").sum())).reset_index().rename(columns={"topic": "Topic"})
    g["Share %"] = (g.Feedback / len(df) * 100).round(1); g["Negative %"] = (g.Negative / g.Feedback * 100).round(1); return g.sort_values("Feedback", ascending=False).reset_index(drop=True)

NO_ANSWER = "I don't have enough information in the current dataset to answer that."
def ask_data(df, q):
    """Rule-based question answering. Returns (answer, supporting table or None)."""
    q = q.lower().strip(); n = len(df); NO = (NO_ANSWER, None)
    if not q or n == 0: return NO
    s = next((x for x in ("positive", "negative", "neutral") if x in q), None)
    if "month" in q and any(k in q for k in ("negative", "worst", "most")):
        d = df[df.sentiment == "Negative"].dropna(subset=["date"])
        if d.empty: return NO
        m = d.groupby(d.date.dt.to_period("M").astype(str)).size().reset_index(); m.columns = ["Month", "Negative feedback"]; t = m.loc[m["Negative feedback"].idxmax()]
        return f"{t['Month']} had the most negative feedback ({int(t['Negative feedback'])} comments).", m
    if "emotion" in q: vc = df.emotion.value_counts(); return f"The most common emotion is {vc.idxmax()} ({vc.iloc[0]} comments, {vc.iloc[0] / n * 100:.1f}%).", vc.rename_axis("Emotion").reset_index(name="Comments")
    if s and any(k in q for k in ("percent", "%", "share", "proportion", "how many", "number", "count")):
        c = int((df.sentiment == s.title()).sum()); return f"{c / n * 100:.1f}% of feedback ({c} of {n} comments) is {s}.", None
    if any(k in q for k in ("improv", "over time", "trend", "getting")):
        t = trend(df); return (NO if t is None else (f"Negative share moved from {t[0]:.1f}% (first half) to {t[1]:.1f}% (second half): sentiment is {t[2]}.", None))
    if any(k in q for k in ("complain", "issue", "problem", "common")):
        t = top_issues(df)
        if t.empty: return "There is no negative feedback in this dataset.", None
        r = t.iloc[0]; return f"{r.Topic} is the biggest source of complaints: {r.Complaints} negative comments ({r['Share %']}% of all negative feedback).", t
    tt = topic_table(df); tt = tt[tt.Feedback >= 3]
    if tt.empty: return NO
    if any(k in q for k in ("highest positive", "most positive", "best")):
        r = tt.assign(p=tt.Positive / tt.Feedback).sort_values("p").iloc[-1]; return f"{r.Topic} has the highest positive share ({r.p * 100:.1f}%).", tt
    if any(k in q for k in ("worst", "most negative", "highest negative", "lowest sentiment")):
        r = tt.sort_values("Negative %").iloc[-1]; return f"{r.Topic} has the worst sentiment: {r['Negative %']}% negative ({r.Negative} comments).", tt
    if "topic" in q or "categor" in q: return f"The most common topic is {tt.iloc[0].Topic} ({tt.iloc[0].Feedback} comments).", tt
    return NO

def ask(df, q): return ask_data(df, q)[0]

def exec_summary(df):
    """Deterministic executive summary: list of (heading, text). Every figure is computed from df."""
    n = len(df); vc = df.sentiment.value_counts(); top = vc.idxmax(); share = vc[top] / n * 100
    overall = f"Overall sentiment is predominantly {top.lower()} ({share:.1f}% of {n} records)." if share >= 50 else f"Overall sentiment is mixed: {', '.join(f'{k.lower()} {v / n * 100:.1f}%' for k, v in vc.items())}."
    out = [("Overall sentiment", overall)]; ti = top_issues(df); tt = topic_table(df); tt = tt[tt.Feedback >= 5]
    if len(ti): out.append(("Most common issue", f"{ti.iloc[0].Topic} ({ti.iloc[0].Complaints} complaints, {ti.iloc[0]['Share %']}% of negative feedback)."))
    t = trend(df); out.append(("Important trend", f"Negative feedback moved from {t[0]:.1f}% to {t[1]:.1f}% between the first and second half of the period ({t[2]})." if t else "No usable dates, so no trend can be reported."))
    if len(tt):
        w, b = tt.sort_values("Negative %").iloc[-1], tt.assign(p=tt.Positive / tt.Feedback * 100).sort_values("p").iloc[-1]
        out += [("Biggest concern", f"{w.Topic} has the highest negative share ({w['Negative %']}%)."), ("Positive finding", f"{b.Topic} has the highest positive share ({b.p:.1f}%).")]
    if len(ti): out.append(("Recommended focus", RECS.get(ti.iloc[0].Topic, "Investigate recurring complaints.")))
    return out

RECS = {"Customer Service": "Consider investigating customer-service response and waiting times.",
        "Technology": "Investigate recurring application or system problems.",
        "Delivery": "Review delivery timelines and identify common causes of delays.",
        "Pricing": "Review pricing and fee transparency.", "Quality": "Inspect product quality control and returns.",
        "Support": "Review support response and resolution times.", "Staff": "Consider staff training and feedback sessions.",
        "Product": "Review product features against customer expectations."}
RECS.update({"Product Quality": "Inspect product quality control and review defect reports.", "Returns/Refunds": "Review refund processing times and the returns procedure.", "Website/App": "Investigate recurring website or app problems and test the checkout flow.", "Payment": "Review payment failures and billing disputes.", "Other": "Review uncategorised complaints for new themes."})
def recommendations(df, thr=0.30):
    g = df.groupby("topic").agg(n=("sentiment", "size"), neg=("sentiment", lambda s: (s == "Negative").mean())); g = g[g.n >= 5].sort_values("neg", ascending=False)
    out = [f"{RECS.get(t, 'Investigate recurring complaints.')} ({t}: {r.neg * 100:.1f}% negative, {int(r.n)} records)" for t, r in g.iterrows() if r.neg >= thr][:4]
    return out + ["Monitor sentiment trends regularly and review a sample of predictions manually."]

def prioritised_recs(df):
    """HIGH / MEDIUM / LOW actions from each topic's negative share relative to the overall share."""
    overall = (df.sentiment == "Negative").mean() * 100; tt = topic_table(df); tt = tt[tt.Feedback >= 5].sort_values("Negative %", ascending=False); out = []
    for _, r in tt.iterrows():
        neg = r["Negative %"]; pr = "HIGH" if neg >= overall + 10 and neg >= 25 else "MEDIUM" if neg >= overall or neg >= 20 else "LOW"
        out.append(dict(priority=pr, topic=r.Topic, evidence=f"{neg}% negative ({r.Negative} of {r.Feedback} comments) versus {overall:.1f}% overall.", action=RECS.get(r.Topic, "Investigate recurring complaints.") if pr != "LOW" else f"Continue monitoring {str(r.Topic).lower()} feedback."))
    return out

def explain_model(R):
    m = R["metrics"][R["best"]]; basis = "pseudo-labels from a keyword rule, so they show agreement with that rule, not with human judgement" if R["pseudo"] else "the labels provided in the dataset"
    return (f"On {R['ntest']} unseen test records, {R['best']} classified about {m['Accuracy']:.0f}% correctly (accuracy). Of the records it labelled as a class, about {m['Precision']:.0f}% were right (precision); of the records truly in a class, it found about {m['Recall']:.0f}% (recall). F1 ({m['F1']:.0f}%) balances the two. "
            f"These figures are measured against {basis}, and depend on dataset size and quality.")

def report_md(R, df, name):
    from src.report_generator import build_report
    return build_report(R, df, name)
