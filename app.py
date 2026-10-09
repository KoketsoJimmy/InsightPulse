"""InsightPulse - Customer Feedback Intelligence. Run: streamlit run app.py"""
import streamlit as st
import views as v

st.set_page_config(page_title="InsightPulse", page_icon="📈", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1200px}[data-testid=stMetricLabel]{opacity:.75}</style>", unsafe_allow_html=True)
P = lambda f, t, i, u=None, d=False: st.Page(f, title=t, icon=i, url_path=u or f.__name__, default=d)
pages = {"Home": [P(v.dashboard, "Dashboard", "🏠", d=True)],
         "Data": [P(v.upload, "Upload Dataset", "📤"), P(v.quality, "Data Quality", "✅")],
         "Analytics": [P(v.sentiment, "Sentiment Analysis", "💬"), P(v.topics, "Topics & Trends", "📊"), P(v.explorer, "Feedback Explorer", "🔎")],
         "AI Insights": [P(v.summary, "Executive Summary", "🧠"), P(v.ask, "Ask the Data", "❓"), P(v.recommendations, "Recommendations", "🎯")],
         "Machine Learning": [P(v.performance, "Model Performance", "📈"), P(v.comparison, "Model Comparison", "⚖️")],
         "Reports": [P(v.report, "Generate Report", "📄")],
         "About": [P(v.about, "About InsightPulse", "ℹ️")]}
st.navigation(pages).run()
