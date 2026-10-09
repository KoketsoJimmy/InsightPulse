import sys, os, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import data_processing as dp, sentiment_analysis as sa, insights as ins, report_generator as rg
from streamlit.testing.v1 import AppTest

@pytest.fixture(scope="module")
def R(): return sa.run(dp.demo_data(), "text", "sentiment")

def test_demo_dataset():
    d = dp.demo_data(); assert len(d) >= 200 and set(d.sentiment) == {"Positive", "Neutral", "Negative"}
def test_quality_score_dynamic():
    good = dp.quality_report(dp.demo_data(), "text", "sentiment"); assert good["score"] >= 85 and good["duplicates"] == 0
    bad = dp.demo_data().head(40).copy(); bad.loc[:5, "text"] = None; bad.loc[6:9, "sentiment"] = "???"; bad = pd.concat([bad, bad.head(5)])
    q = dp.quality_report(bad, "text", "sentiment"); assert q["score"] < good["score"] and q["empty"] > 0 and q["invalid_sentiment"] > 0 and q["duplicates"] == 5
def test_four_models_same_split(R): assert set(R["metrics"]) == {"Naive Bayes", "Logistic Regression", "Random Forest", "Linear SVM"} and R["cm"].sum() == R["ntest"]
def test_topics():
    assert sa.topic("Delivery was extremely late") == "Delivery" and sa.topic("My refund has not arrived") == "Returns/Refunds" and sa.topic("hello") == "Other"
def test_pseudo_labels_flagged():
    r = sa.run(dp.demo_data().drop(columns="sentiment"), "text", None); assert r["pseudo"] and "Pseudo" in r["label_source"]
def test_invalid_labels_excluded():
    d = dp.demo_data(); d.loc[:9, "sentiment"] = "unknown"; r = sa.run(d, "text", "sentiment"); assert r["invalid_dropped"] == 10 and not r["pseudo"]
def test_ask_calculated_and_fallback(R):
    df = R["df"]; a, t = ins.ask_data(df, "What percentage of feedback is negative?"); assert f"{(df.sentiment == 'Negative').mean() * 100:.1f}%" in a
    assert ins.ask_data(df, "Who is the CEO?")[0] == ins.NO_ANSWER
    a, t = ins.ask_data(df, "Which month had the most negative feedback?"); assert t is not None and "most negative" in a
    assert ins.ask_data(df.assign(date=pd.NaT), "Which month had the most negative feedback?")[0] == ins.NO_ANSWER
def test_summary_and_recs(R):
    assert any(h == "Overall sentiment" for h, _ in ins.exec_summary(R["df"])); p = ins.prioritised_recs(R["df"]); assert p and p[0]["priority"] in ("HIGH", "MEDIUM", "LOW")
def test_report_sections(R):
    md = rg.build_report(R, R["df"], "x"); [s for s in ("Dataset overview", "Data quality", "Model comparison", "Executive summary", "Recommendations", "limitations") if s.lower() not in md.lower() and pytest.fail(s)]
    assert "<table>" in rg.to_html(md)
@pytest.mark.parametrize("fn", ["dashboard", "upload", "quality", "sentiment", "topics", "explorer", "summary", "ask", "recommendations", "performance", "comparison", "report", "about"])
def test_pages_load(fn):
    at = AppTest.from_string(f"import views\nviews.load_demo()\nviews.{fn}()", default_timeout=90).run(); assert not at.exception
def test_empty_state_and_app_start():
    assert not AppTest.from_string("import views\nviews.dashboard()").run().exception; assert not AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"), default_timeout=60).run().exception
