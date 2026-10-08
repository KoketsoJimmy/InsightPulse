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
