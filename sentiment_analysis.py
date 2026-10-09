"""TF-IDF + Naive Bayes / Logistic Regression, emotion and category rules."""
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.data_processing import clean, tokens, load_clean

TOPICS = {  # easy to extend: add a topic and its keywords
 "Delivery": "delivery delivered shipping courier late parcel package tracking dispatch delayed",
 "Customer Service": "service agent rude helpdesk support representative staff polite queue call calls",
 "Product Quality": "quality broken faulty defective durable product item damaged",
 "Pricing": "price prices pricing expensive cheap cost value fees fee overpriced discount",
 "Returns/Refunds": "return returns refund refunded exchange replacement",
 "Website/App": "website app application site login crashing crashes crash checkout slow update",
 "Payment": "payment card charged billing declined invoice",
}
TOPICS = {k: set(v.split()) for k, v in TOPICS.items()}
EMOTIONS = {
 "Anger": "angry furious unacceptable outrageous rude terrible worst disgusting hate",
 "Frustration": "waited waiting crash crashes crashing slow keeps annoying frustrating confusing broken late delay",
 "Sadness": "disappointed disappointing sad unhappy sorry upset let",
 "Happiness": "love amazing excellent fantastic great wonderful delighted brilliant",
 "Satisfaction": "good helpful easy smooth friendly happy satisfied pleased recommend useful fast",
}
EMOTIONS = {k: set(v.split()) for k, v in EMOTIONS.items()}
POS = set("good great excellent amazing helpful easy love fast friendly happy smooth fantastic wonderful recommend useful pleased brilliant".split())
NEG = set("bad terrible worst slow rude crash crashes crashing broken late waited disappointed disappointing awful expensive confusing frustrating unacceptable faulty".split())

def lexicon_label(t):
    w = tokens(t); s = sum(x in POS for x in w) - sum(x in NEG for x in w)
    return "Positive" if s > 0 else "Negative" if s < 0 else "Neutral"

def emotion(t, sent="Neutral"):
    w = set(tokens(t)); sc = {k: len(w & v) for k, v in EMOTIONS.items()}
    best = max(sc, key=sc.get)
    if sc[best] == 0: return {"Positive": "Satisfaction", "Negative": "Frustration"}.get(sent, "Neutral")
    return best

def topic(t):
    w = set(tokens(t)); sc = {k: len(w & v) for k, v in TOPICS.items()}
    best = max(sc, key=sc.get); return best if sc[best] else "Other"


def score_of(model, P):  # sentiment score in [-1, 1] = P(positive) - P(negative)
    c = list(model.classes_); return P[:, c.index("Positive")] - P[:, c.index("Negative")]

def run(raw, col, sentiment_col="auto"):
    """Pipeline: clean -> label -> train/compare -> predict -> emotion/topic.
    sentiment_col: 'auto' (use a 'sentiment' column if present), a column name, or None (pseudo-labels)."""
    from src.model_training import compare_models
    from src.data_processing import quality_report, VALID
    df = load_clean(raw, col); sc = "sentiment" if sentiment_col == "auto" and "sentiment" in df.columns else (sentiment_col if sentiment_col not in (None, "auto") and sentiment_col in df.columns else None)
    quality = quality_report(raw, col, sc); dropped = 0
    if sc:
        lab = df[sc].astype(str).str.strip().str.title(); ok = lab.isin(VALID)
        if ok.mean() >= .5: dropped = int((~ok).sum()); df = df[ok].copy(); df["label"] = lab[ok]
        else: sc = None
    pseudo = sc is None
    if pseudo: df["label"] = df["text"].map(lexicon_label)
    src = "Pseudo-labelled (keyword rule, not human-verified)" if pseudo else f"Provided labels ('{sc}')"
    if len(df) < 30 or df["label"].value_counts().min() < 4: raise ValueError("Need at least 30 rows with 4+ examples per sentiment class to train models.")
    M = compare_models(df["text"], df["label"]); model = M["models"][M["best"]]
    P = model.predict_proba(M["vec"].transform(df["clean_text"]))
    df["sentiment"] = model.classes_[P.argmax(1)]; df["confidence"] = (P.max(1) * 100).round(1); df["score"] = score_of(model, P).round(3)
    df["emotion"] = [emotion(t, s) for t, s in zip(df["text"], df["sentiment"])]
    df["topic"] = df["category"].astype(str) if "category" in df.columns else df["text"].map(topic)
    dc = next((c for c in df.columns if str(c).lower() == "date"), None); df["date"] = pd.to_datetime(df[dc], errors="coerce") if dc else pd.NaT
    rc = next((c for c in df.columns if str(c).lower() == "rating"), None); df["rating"] = pd.to_numeric(df[rc], errors="coerce") if rc else float("nan")
    joblib.dump({"model": model, "vectorizer": M["vec"]}, "models/sentiment_model.pkl")
    return dict(df=df, vec=M["vec"], model=model, models=M["models"], best=M["best"], metrics=M["metrics"], notes=M["notes"], cm=M["cm"], cm_labels=M["cm_labels"],
                ntrain=M["ntrain"], ntest=M["ntest"], label_source=src, pseudo=pseudo, invalid_dropped=dropped, quality=quality, col=col)

def analyse_one(R, text):
    c = clean(text)
    if not c: raise ValueError("Please enter some text to analyse.")
    P = R["model"].predict_proba(R["vec"].transform([c])); s = R["model"].classes_[P[0].argmax()]
    e, t = emotion(text, s), topic(text); hits = [w for w in tokens(text) if w in POS | NEG]
    why = f"Classified {s.lower()} by {R['best']}" + (f"; influential words: {', '.join(hits[:4])}." if hits else ".") + f" Emotion '{e}' comes from keyword rules; category '{t}' from category keywords."
    return {"sentiment": s, "confidence": round(float(P.max()) * 100, 1), "score": round(float(score_of(R["model"], P)[0]), 2), "emotion": e, "topic": t, "explanation": why}
