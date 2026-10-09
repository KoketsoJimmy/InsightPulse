"""Streamlit pages for InsightPulse (each public function is one sidebar page)."""
import pandas as pd, streamlit as st
from src import data_processing as dp, sentiment_analysis as sa, insights as ins, visualizations as viz, report_generator as rg

NONE = "(none - generate pseudo-labels)"
ICON = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}
EXAMPLES = ["What are customers complaining about the most?", "What percentage of feedback is negative?", "Which topic has the most complaints?", "Has sentiment improved over time?", "Which month had the most negative feedback?", "What is the most common emotion?", "Which topic has the worst sentiment?"]

# ---------- helpers ----------
def process(raw: pd.DataFrame, name: str, col=None, sent="auto") -> None:
    col = col or dp.detect_text_col(raw)
    if col is None: raise ValueError("The dataset does not contain a valid text column.")
    with st.spinner("Cleaning data and training models..."): R = sa.run(raw, col, sent)
    st.session_state.update(R=R, raw=raw, name=name)

def load_demo() -> None: process(dp.demo_data(), "InsightPulse demo dataset (SYNTHETIC)", "text", "sentiment")

def need_data() -> dict:
    R = st.session_state.get("R")
    if R is None:
        st.info("No dataset loaded yet. Load the built-in synthetic demo or upload your own CSV on the **Upload Dataset** page.")
        if st.button("Load Demo Dataset", type="primary", key="empty_demo"):
            try: load_demo(); st.rerun()
            except Exception as e: st.error(f"Could not load the demo: {e}")
        st.stop()
    return R

def badge(R: dict) -> None:
    if R["pseudo"]: st.warning("**Pseudo-labelled.** Sentiment labels come from a keyword rule and are not human-verified. Metrics measure agreement with that rule, not real-world accuracy.")
    else: st.caption(f"Labels: {R['label_source']}")
    if "SYNTHETIC" in st.session_state.get("name", ""): st.caption("Synthetic demo data: results do not represent real-world performance.")

def kpis(items) -> None:
    for c, (label, value) in zip(st.columns(len(items)), items):
        with c, st.container(border=True): st.metric(label, value)

def show(fig, msg="Not enough data (for example, no dates) to draw this chart.") -> None:
    st.plotly_chart(fig, width="stretch") if fig is not None else st.info(msg)

def filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    with st.expander("Filters", expanded=True):
        a, b, c = st.columns(3); all_ = lambda x: x or "All"
        f = {"q": a.text_input("Search keyword", key=key + "q"), "sentiment": b.selectbox("Sentiment", [""] + sorted(df.sentiment.unique()), format_func=all_, key=key + "s"), "topic": c.selectbox("Topic", [""] + sorted(df.topic.unique()), format_func=all_, key=key + "t")}
        d, e = st.columns(2)
        if df.date.notna().any():
            lo, hi = df.date.min().date(), df.date.max().date(); r = d.date_input("Date range", (lo, hi), min_value=lo, max_value=hi, key=key + "d")
            if len(r) == 2: f["start"], f["end"] = pd.Timestamp(r[0]), pd.Timestamp(r[1])
        out = ins.filt(df, f)
        if df.rating.notna().any() and int(df.rating.min()) < int(df.rating.max()):
            lo, hi = int(df.rating.min()), int(df.rating.max()); rr = e.slider("Rating", lo, hi, (lo, hi), key=key + "r"); out = out[out.rating.between(*rr)]
    if out.empty: st.warning("No records match these filters. Try widening them."); st.stop()
    return out

def show_quality(q: dict) -> None:
    colour = {"good": "green", "warn": "orange", "bad": "red"}[q["level"]]
    st.markdown(f"### Data Quality Score: :{colour}[{q['score']}/100]")
    for lvl, msg in q["checks"]: st.markdown({"good": "✅", "warn": "⚠️", "bad": "❌"}[lvl] + " " + msg)
    kpis([("Rows", q["rows"]), ("Columns", q["columns"]), ("Missing cells", q["missing"]), ("Duplicates", q["duplicates"])])
    kpis([("Empty feedback", q["empty"]), ("Invalid sentiment", "n/a" if q["invalid_sentiment"] is None else q["invalid_sentiment"]), ("Avg length (words)", q["avg_words"]), ("Unique feedback", q["unique_text"])])

def read_csv(up):
    if up.size > 5 * 1024 * 1024: return None, "File is larger than 5 MB."
    try: raw = pd.read_csv(up)
    except UnicodeDecodeError:
        up.seek(0)
        try: raw = pd.read_csv(up, encoding="latin-1")
        except Exception: return None, "This file could not be read as a CSV."
    except pd.errors.EmptyDataError: return None, "The uploaded dataset is empty."
    except Exception: return None, "This file is not a valid CSV."
    return (None, "The uploaded dataset is empty.") if raw.empty else (raw, None)

# ---------- pages ----------
def dashboard() -> None:
    st.title("InsightPulse"); st.subheader("Customer Feedback Intelligence")
    st.write("Turn customer feedback into actionable insights using data analytics, machine learning and AI.")
    R = need_data(); df = R["df"]; badge(R); pc = lambda s: f"{(df.sentiment == s).mean() * 100:.1f}%"
    kpis([("Total Feedback", len(df)), ("Positive", pc("Positive")), ("Neutral", pc("Neutral")), ("Negative", pc("Negative")), ("Model Accuracy", f"{R['metrics'][R['best']]['Accuracy']}%")])
    a, b = st.columns(2)
    with a: show(viz.sentiment_dist(df))
    with b: show(viz.top_issues(df), "No negative feedback to show.")
    c, d = st.columns(2)
    with c: show(viz.over_time(df))
    with d: show(viz.pos_neg_trend(df))
    show(viz.volume(df))

def upload() -> None:
    st.header("Upload Dataset"); st.caption("Files are processed locally and never sent anywhere.")
    c1, c2 = st.columns(2)
    if c1.button("Load Demo Dataset", type="primary"):
        try: load_demo(); st.success("Demo dataset loaded. Open the Dashboard.")
        except Exception as e: st.error(str(e))
    c2.download_button("Download Sample Dataset", dp.demo_data().to_csv(index=False), "insightpulse_sample.csv", "text/csv")
    st.markdown("**Step 1: Upload dataset (CSV)**"); up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if up is not None:
        raw, err = read_csv(up)
        if err: st.error(err)
        else:
            cols = list(raw.columns); g = dp.detect_text_col(raw)
            st.markdown("**Step 2: Select feedback column**"); col = st.selectbox("Feedback column", cols, index=cols.index(g) if g in cols else 0)
            st.markdown("**Step 3: Select sentiment column (optional)**"); lower = {str(c).lower(): c for c in cols}; opts = [NONE] + cols
            sent = st.selectbox("Sentiment column", opts, index=opts.index(lower["sentiment"]) if "sentiment" in lower else 0); sc = None if sent == NONE else sent
            st.markdown("**Step 4: Validate dataset**"); show_quality(dp.quality_report(raw, col, sc))
            if sc is None: st.warning("No sentiment column selected. Results will be **pseudo-labelled** by a transparent keyword rule, not human-verified.")
            st.markdown("**Step 5: Analyse dataset**")
            if st.button("Analyse Dataset", type="primary"):
                try: process(raw, up.name, col, sc); st.success("Analysis complete. Explore the other pages.")
                except ValueError as e: st.error(str(e))
                except Exception: st.error("Unable to analyse this dataset. Check the selected columns.")
    if "R" in st.session_state:
        st.divider(); st.subheader(f"Loaded: {st.session_state['name']}"); badge(st.session_state["R"]); st.dataframe(st.session_state["raw"].head(20), width="stretch")

def quality() -> None:
    st.header("Data Quality"); R = need_data(); show_quality(R["quality"])
    if R["invalid_dropped"]: st.warning(f"{R['invalid_dropped']} records with invalid sentiment labels were excluded from training.")

def sentiment() -> None:
    st.header("Sentiment Analysis"); R = need_data(); badge(R); d = filters(R["df"], "sa_"); show(viz.sentiment_dist(d))
    v = d[["text", "sentiment", "confidence", "topic", "emotion"]].copy(); v["sentiment"] = v.sentiment.map(lambda s: f"{ICON[s]} {s}")
    st.dataframe(v.rename(columns={"text": "Feedback", "sentiment": "Sentiment", "confidence": "Confidence %", "topic": "Topic", "emotion": "Emotion"}), width="stretch")

def topics() -> None:
    st.header("Topics & Trends"); R = need_data(); df = R["df"]; badge(R); st.caption("Topics come from transparent keyword rules (or a 'category' column if your CSV has one).")
    a, b = st.columns(2)
    with a: show(viz.top_issues(df), "No negative feedback to show.")
    with b: st.dataframe(ins.topic_table(df), width="stretch", hide_index=True)
    sel = st.selectbox("View feedback for a topic", sorted(df.topic.unique())); st.dataframe(df[df.topic == sel][["text", "sentiment", "confidence"]], width="stretch")
    c, d = st.columns(2)
    with c: show(viz.volume(df))
    with d: show(viz.pos_neg_trend(df))

def explorer() -> None:
    st.header("Feedback Explorer"); R = need_data(); d = filters(R["df"], "ex_")
    by = st.selectbox("Sort by", ["date", "confidence", "sentiment", "topic"]); d = d.sort_values(by, ascending=by == "sentiment")
    st.dataframe(d[["text", "sentiment", "confidence", "topic", "date", "rating"]], width="stretch"); st.caption(f"{len(d)} records")
    i = st.selectbox("View individual feedback", d.index, format_func=lambda i: str(d.loc[i, "text"])[:70])
    r = d.loc[i]
    with st.container(border=True): st.markdown(f"**Feedback:** \"{r.text}\"\n\n**Sentiment:** {ICON[r.sentiment]} {r.sentiment} ({r.confidence}% confidence)\n\n**Topic:** {r.topic}  \n**Emotion:** {r.emotion}")

def summary() -> None:
    st.header("AI Executive Summary"); R = need_data(); badge(R)
    with st.container(border=True):
        for h, t in ins.exec_summary(R["df"]): st.markdown(f"**{h}:** {t}")
    st.caption("Generated by deterministic Python rules from the loaded dataset. No external AI model is used.")

def ask() -> None:
    st.header("Ask the Data"); R = need_data(); ex = st.selectbox("Example questions", [""] + EXAMPLES, format_func=lambda x: x or "Choose an example...")
    q = st.text_input("Or type your own question", value=ex)
    if q:
        a, t = ins.ask_data(R["df"], q); st.markdown(f"**Answer:** {a}")
        if t is not None: st.caption("Supporting data"); st.dataframe(t, width="stretch", hide_index=True)

def recommendations() -> None:
    st.header("Recommendations"); R = need_data(); badge(R); recs = ins.prioritised_recs(R["df"])
    if not recs: st.info("Not enough records per topic (at least 5) to make recommendations.")
    for r in recs: {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.info}[r["priority"]](f"**{r['priority']} PRIORITY: {r['topic']}**\n\nEvidence: {r['evidence']}\n\nRecommendation: {r['action']}")

def performance() -> None:
    st.header("Model Performance"); R = need_data(); badge(R); m = R["metrics"][R["best"]]
    st.write(f"Best model: **{R['best']}** (trained on {R['ntrain']} records, evaluated on {R['ntest']} unseen test records).")
    kpis([("Accuracy", f"{m['Accuracy']}%"), ("Precision", f"{m['Precision']}%"), ("Recall", f"{m['Recall']}%"), ("F1 Score", f"{m['F1']}%")])
    st.info(ins.explain_model(R)); show(viz.confusion(R))
    with st.expander("What do these metrics mean?"): st.markdown("- **Accuracy**: share of predictions that were correct.\n- **Precision**: of items predicted as a class, how many truly were.\n- **Recall**: of items truly in a class, how many were found.\n- **F1**: balance of precision and recall (weighted across classes).")

def comparison() -> None:
    st.header("Model Comparison"); R = need_data(); badge(R); st.caption("All models use the same TF-IDF features and the same train/test split.")
    st.dataframe(pd.DataFrame(R["metrics"]).T.rename_axis("Model").reset_index(), width="stretch", hide_index=True)
    for k, v in R["notes"].items(): st.warning(f"{k}: {v}")
    st.success(f"BEST MODEL: {R['best']} (F1 {R['metrics'][R['best']]['F1']}%). Ties go to the simpler model."); show(viz.model_comparison(R))

def report() -> None:
    st.header("Generate Report"); R = need_data(); md = rg.build_report(R, R["df"], st.session_state["name"])
    c1, c2 = st.columns(2); c1.download_button("Download Report (HTML)", rg.to_html(md), "insightpulse_report.html", "text/html", type="primary"); c2.download_button("Download Report (Markdown)", md, "insightpulse_report.md", "text/markdown")
    st.caption("Open the HTML report in a browser and use Print > Save as PDF if you need a PDF."); st.divider(); st.markdown(md)

def about() -> None:
    st.header("About InsightPulse"); st.write("InsightPulse is an educational Python + Streamlit prototype: pandas, NLTK, scikit-learn and Plotly, running fully offline.")
    with st.container(border=True):
        st.subheader("Data & AI Notice"); st.markdown("- Synthetic/demo data may be used and does not represent real customers.\n- Automatically generated labels may be **pseudo-labels**, not human-verified ground truth.\n- Model performance depends on dataset quality and size.\n- Topics and emotions use simple keyword rules; insights are rule-based, not a language model.\n- This is a demonstration tool and should not drive decisions without validation and human oversight.")
    with st.expander("Presenter guide", expanded=True): st.markdown("1. Upload Dataset > **Load Demo Dataset**\n2. Dashboard\n3. Sentiment Analysis\n4. Topics & Trends (top issues)\n5. Ask the Data\n6. Model Performance and Model Comparison\n7. Recommendations\n8. Generate Report > Download")
