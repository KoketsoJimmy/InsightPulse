"""Plotly charts."""
import pandas as pd, plotly.express as px
from src.data_processing import tokens
CM = {"Positive": "#2ecc9a", "Neutral": "#e0b84c", "Negative": "#ff6b6b"}
def _t(f): f.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text=""); return f
def sentiment_dist(df): return _t(px.pie(df, names="sentiment", color="sentiment", color_discrete_map=CM, hole=.45, title="Sentiment distribution"))
def by_category(df): return _t(px.histogram(df, x="topic", color="sentiment", barmode="stack", color_discrete_map=CM, title="Sentiment by category"))
def over_time(df):
    d = df.dropna(subset=["date"])
    if d.empty: return None
    m = d.groupby([d.date.dt.to_period("M").astype(str), "sentiment"]).size().reset_index(name="count").rename(columns={"date": "month"})
    return _t(px.line(m, x="month", y="count", color="sentiment", markers=True, color_discrete_map=CM, title="Sentiment over time"))
def emotions(df): return _t(px.histogram(df, x="emotion", color="emotion", title="Emotion distribution"))
def keywords(df, n=12):
    s = pd.Series([w for t in df.text for w in tokens(t)]).value_counts().head(n).reset_index(); s.columns = ["word", "count"]
    return _t(px.bar(s.iloc[::-1], x="count", y="word", orientation="h", title="Top keywords"))
def pos_vs_neg(df):
    g = pd.crosstab(df.topic, df.sentiment).reindex(columns=["Positive", "Negative"], fill_value=0).reset_index().melt("topic", var_name="sentiment", value_name="count")
    return _t(px.bar(g, x="topic", y="count", color="sentiment", barmode="group", color_discrete_map=CM, title="Positive vs negative categories"))

from src import insights as _ins
def _monthly(df):
    d = df.dropna(subset=["date"]); return None if d.empty else d.assign(month=d.date.dt.to_period("M").astype(str))
def volume(df):
    m = _monthly(df); return None if m is None else _t(px.bar(m.groupby("month").size().reset_index(name="count"), x="month", y="count", title="Feedback volume over time"))
def pos_neg_trend(df):
    m = _monthly(df)
    if m is None: return None
    g = m.groupby("month").sentiment.value_counts(normalize=True).mul(100).round(1).unstack(fill_value=0).reindex(columns=["Positive", "Negative"], fill_value=0).reset_index().melt("month", var_name="sentiment", value_name="percent")
    return _t(px.line(g, x="month", y="percent", color="sentiment", markers=True, color_discrete_map=CM, title="Positive vs negative trend (%)"))
def top_issues(df):
    t = _ins.top_issues(df); return None if t.empty else _t(px.bar(t.iloc[::-1], x="Complaints", y="Topic", orientation="h", text="Share %", title="Top customer issues (negative feedback)"))
def confusion(R): return _t(px.imshow(R["cm"], x=R["cm_labels"], y=R["cm_labels"], text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"), title=f"Confusion matrix: {R['best']}"))
def model_comparison(R):
    m = pd.DataFrame(R["metrics"]).T.reset_index().rename(columns={"index": "Model"}).melt("Model", var_name="Metric", value_name="Percent")
    return _t(px.bar(m, x="Model", y="Percent", color="Metric", barmode="group", title="Model comparison (test set)"))
