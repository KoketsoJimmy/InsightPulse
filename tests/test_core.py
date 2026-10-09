import sys, os, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import data_processing as dp, sentiment_analysis as sa, insights as ins, visualizations as viz
@pytest.fixture(scope="module")
def R(): return sa.run(dp.sample_data(), "text")
def test_clean(): assert dp.clean("Visit http://x.com NOW!!  ok") == "visit now ok"
def test_tokens(): assert "the" not in dp.tokens("the app crashes")
def test_missing_dupes():
    df = pd.DataFrame({"text": ["a good day"] * 3 + [None, ""]}); assert dp.data_summary(df, "text")["duplicates"] == 2
    assert len(dp.load_clean(df, "text")) == 1
def test_empty():
    with pytest.raises(ValueError): dp.load_clean(pd.DataFrame({"text": [None]}), "text")
def test_no_text_col(): assert dp.detect_text_col(pd.DataFrame({"a": [1], "b": [2]})) is None
def test_too_small():
    with pytest.raises(ValueError): sa.run(pd.DataFrame({"text": ["good", "bad"]}), "text")
def test_metrics(R): assert R["best"] in R["metrics"] and all(0 <= v <= 100 for m in R["metrics"].values() for v in m.values())
def test_predict(R): r = sa.analyse_one(R, "The service was amazing."); assert r["sentiment"] == "Positive" and -1 <= r["score"] <= 1
def test_kpis(R): d = R["df"]; assert abs(sum((d.sentiment == s).mean() for s in ["Positive", "Neutral", "Negative"]) - 1) < 1e-9
def test_filter(R): assert (ins.filt(R["df"], {"sentiment": "Negative"}).sentiment == "Negative").all()
def test_ask(R): assert "negative" in ins.ask(R["df"], "How many comments are negative?")
def test_report(R): assert "## Recommendations" in ins.report_md(R, R["df"], "x")
def test_charts(R): assert all(f(R["df"]) is not None for f in (viz.sentiment_dist, viz.by_category, viz.over_time, viz.emotions, viz.keywords, viz.pos_vs_neg))
