"""InsightPulse - Streamlit dashboard. Run: streamlit run app.py"""
import pandas as pd, streamlit as st
from src import data_processing as dp, sentiment_analysis as sa, insights as ins, visualizations as viz

st.set_page_config(page_title="InsightPulse", page_icon="📈", layout="wide")
st.title("InsightPulse")
st.caption("Turn feedback into actionable insights.")

def process(raw, name, col=None):
    col = col or dp.detect_text_col(raw)
    if col is None: raise ValueError("The dataset does not contain a valid text column.")
    with st.spinner("Cleaning data and training models..."):
        st.session_state.update(R=sa.run(raw, col), raw=raw, name=name, summary=dp.data_summary(raw, col))

page = st.sidebar.radio("Navigate", ["Dataset", "Dashboard", "Analyse Feedback", "Model Performance", "AI Insights", "About"])
R, df = st.session_state.get("R"), None
if R is not None: df = R["df"]
if page in ("Dashboard", "Analyse Feedback", "Model Performance", "AI Insights") and R is None:
    st.info("No dataset loaded yet. Go to **Dataset** and upload a CSV or load the synthetic sample."); st.stop()

if page == "Dataset":
    c1, c2 = st.columns(2)
    if c1.button("Load synthetic sample dataset", type="primary"):
        try: process(dp.sample_data(), "sample_feedback.csv (SYNTHETIC DATA)"); st.success("Sample loaded.")
        except Exception as e: st.error(str(e))
    up = c2.file_uploader("Upload a CSV (processed locally, never sent anywhere)", type=["csv"])
    if up is not None:
        try:
            if up.size > 5 * 1024 * 1024: raise ValueError("File is larger than 5 MB.")
            raw = pd.read_csv(up)
            if raw.empty: raise ValueError("The uploaded dataset is empty.")
            guess = dp.detect_text_col(raw)
            col = st.selectbox("Feedback text column", list(raw.columns), index=list(raw.columns).index(guess) if guess in raw.columns else 0)
            if st.button("Analyse this file"): process(raw, up.name, col); st.success("Analysis complete.")
        except pd.errors.EmptyDataError: st.error("The uploaded dataset is empty.")
        except ValueError as e: st.error(str(e))
        except Exception: st.error("Unable to read this file as a CSV.")
    if "summary" in st.session_state:
        s = st.session_state["summary"]; st.subheader(st.session_state["name"])
        a, b, c, d = st.columns(4); a.metric("Records", s["records"]); b.metric("Columns", s["columns"]); c.metric("Missing text", s["missing"]); d.metric("Duplicates", s["duplicates"])
        st.write(f"Text column: **{st.session_state['R']['col']}**. Labels: {st.session_state['R']['label_source']}."); st.dataframe(st.session_state["raw"].head(20), width="stretch")

elif page == "Dashboard":
    sb = st.sidebar; sb.subheader("Filters")
    f = {"q": sb.text_input("Keyword search")}
    for k in ("sentiment", "emotion", "topic"): f[k] = sb.selectbox(k.title() if k != "topic" else "Category", [""] + sorted(df[k].unique()), format_func=lambda x: x or "All")
    if df.date.notna().any():
        lo, hi = df.date.min().date(), df.date.max().date(); r = sb.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
        if len(r) == 2: f["start"], f["end"] = pd.Timestamp(r[0]), pd.Timestamp(r[1])
    d = ins.filt(df, f)
    if d.empty: st.warning("No records match these filters."); st.stop()
    n = len(d); k = st.columns(6); pc = lambda s: f"{(d.sentiment == s).mean() * 100:.1f}%"
    for col, (l, v) in zip(k, [("Total feedback", n), ("Positive", pc("Positive")), ("Neutral", pc("Neutral")), ("Negative", pc("Negative")), ("Top category", d.topic.value_counts().idxmax()), ("Avg score", f"{d.score.mean():.2f}")]): col.metric(l, v)
    t = st.tabs(["Sentiment", "Categories", "Time", "Emotions", "Keywords", "Pos vs Neg"])
    for tab, fig in zip(t, [viz.sentiment_dist(d), viz.by_category(d), viz.over_time(d), viz.emotions(d), viz.keywords(d), viz.pos_vs_neg(d)]):
        with tab:
            st.plotly_chart(fig, width="stretch") if fig else st.info("No date column available.")
    st.subheader("Category analysis")
    g = d.groupby("topic").agg(Records=("text", "size"), Positive=("sentiment", lambda s: round((s == "Positive").mean() * 100, 1)), Negative=("sentiment", lambda s: round((s == "Negative").mean() * 100, 1)), AvgScore=("score", "mean")).round(2)
    st.dataframe(g.rename(columns={"Positive": "Positive %", "Negative": "Negative %", "AvgScore": "Avg score"}), width="stretch")
    st.subheader("Feedback table"); st.dataframe(d[["text", "sentiment", "confidence", "score", "emotion", "topic", "date"]], width="stretch")

elif page == "Analyse Feedback":
    txt = st.text_area("Enter feedback to analyse", placeholder="The application is useful but keeps crashing.", max_chars=2000)
    if st.button("Analyse Feedback", type="primary"):
        try:
            r = sa.analyse_one(R, txt); a, b, c, e, g = st.columns(5)
            a.metric("Sentiment", r["sentiment"]); b.metric("Confidence", f"{r['confidence']}%"); c.metric("Score", r["score"]); e.metric("Emotion", r["emotion"]); g.metric("Category", r["topic"])
            st.info(r["explanation"])
        except ValueError as e: st.error(str(e))

elif page == "Model Performance":
    m = pd.DataFrame(R["metrics"]).T; st.write(f"Trained on {R['ntrain']} records, tested on {R['ntest']} unseen records. Labels: {R['label_source']}.")
    st.dataframe(m.style.format("{:.1f}%"), width="stretch")
    st.success(f"Best model: {R['best']}, chosen for the highest weighted F1 on the test set.")
    with st.expander("What do these metrics mean?", expanded=True):
        st.markdown("- **Accuracy**: share of all predictions that were correct.\n- **Precision**: of the items predicted as a class, how many truly were.\n- **Recall**: of the items truly in a class, how many were found.\n- **F1**: balance (harmonic mean) of precision and recall.")
    st.caption("Scores on the synthetic sample are optimistic because it is generated from templates.")

elif page == "AI Insights":
    i = ins.insights(df); st.subheader("Summary"); st.write(i["summary"])
    c1, c2 = st.columns(2)
    with c1: st.subheader("Key findings"); [st.write("- " + x) for x in i["findings"] + i["negatives"] + i["positives"]]
    with c2: st.subheader("Recommendations"); [st.write(f"{n}. {x}") for n, x in enumerate(ins.recommendations(df), 1)]
    st.subheader("Ask the Data"); q = st.text_input("Ask a question", placeholder="Which category has the worst sentiment?")
    if q: st.write(ins.ask(df, q))
    md = ins.report_md(R, df, st.session_state["name"])
    open("reports/insights_report.md", "w", encoding="utf-8").write(md)
    st.download_button("Download insights report", md, "insights_report.md", "text/markdown")

else:
    st.subheader("About"); st.write("InsightPulse is a CAPACITI Week 3 educational prototype: Python, pandas, NLTK, scikit-learn and Plotly, running locally.")
    with st.expander("Ethics and limitations", expanded=True):
        st.markdown("- Sample data is **synthetic**; uploads stay on your machine.\n- Models can be wrong, can be biased, and struggle with sarcasm and context.\n- Emotion and category use simple keyword rules, not deep learning.\n- Results should support, not replace, human judgement.")
