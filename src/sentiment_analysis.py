"""TF-IDF + Naive Bayes / Logistic Regression, emotion and category rules."""
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.data_processing import clean, tokens, load_clean

TOPICS = {  # easy to extend: add a topic + keywords
 "Customer Service": "service waited wait queue rude helpful reception agent",
 "Technology": "app application crash crashes website login bug slow system software online",
 "Product": "product item feature design",
 "Pricing": "price pricing expensive cheap cost fee value money",
 "Delivery": "delivery delivered shipping courier late parcel arrived package",
 "Support": "support helpdesk ticket response resolved refund",
 "Staff": "staff team employee manager friendly",
 "Quality": "quality durable broken faulty well-made defect",
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
    best = max(sc, key=sc.get); return best if sc[best] else "Product"


def train(texts, labels):
    X = [clean(t) for t in texts]
    xtr, xte, ytr, yte = train_test_split(X, labels, test_size=.25, random_state=42, stratify=labels)
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(xtr); res, models = {}, {}
    for name, m in {"Naive Bayes": MultinomialNB(), "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced")}.items():
        m.fit(vec.transform(xtr), ytr); p = m.predict(vec.transform(xte))
        pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="weighted", zero_division=0)
        res[name] = {k: round(float(v) * 100, 1) for k, v in dict(Accuracy=accuracy_score(yte, p), Precision=pr, Recall=rc, F1=f1).items()}
        models[name] = m
    best = max(res, key=lambda k: res[k]["F1"])
    return vec, models[best], best, res, len(xtr), len(xte)

def score_of(model, P):  # sentiment score in [-1, 1] = P(positive) - P(negative)
    c = list(model.classes_); return P[:, c.index("Positive")] - P[:, c.index("Negative")]

def run(raw, col):
    """Full pipeline: clean -> label -> train -> predict -> emotion/category."""
    df = load_clean(raw, col)
    ok = "sentiment" in df.columns and df["sentiment"].astype(str).str.title().isin(["Positive", "Neutral", "Negative"]).all()
    df["label"] = df["sentiment"].astype(str).str.title() if ok else df["text"].map(lexicon_label)
    src = "Provided 'sentiment' column" if ok else "Keyword-lexicon pseudo-labels (no labelled column)"
    if len(df) < 30 or df["label"].value_counts().min() < 4: raise ValueError("Need at least 30 rows with 4+ examples per sentiment class.")
    vec, model, best, res, ntr, nte = train(df["text"], df["label"])
    P = model.predict_proba(vec.transform(df["clean_text"]))
    df["sentiment"] = model.classes_[P.argmax(1)]; df["confidence"] = (P.max(1) * 100).round(1); df["score"] = score_of(model, P).round(3)
    df["emotion"] = [emotion(t, s) for t, s in zip(df["text"], df["sentiment"])]
    df["topic"] = df["category"].astype(str) if "category" in df.columns else df["text"].map(topic)
    dc = next((c for c in df.columns if str(c).lower() == "date"), None)
    df["date"] = pd.to_datetime(df[dc], errors="coerce") if dc else pd.NaT
    joblib.dump({"model": model, "vectorizer": vec}, "models/sentiment_model.pkl")
    return dict(df=df, vec=vec, model=model, best=best, metrics=res, ntrain=ntr, ntest=nte, label_source=src, col=col)

def analyse_one(R, text):
    c = clean(text)
    if not c: raise ValueError("Please enter some text to analyse.")
    P = R["model"].predict_proba(R["vec"].transform([c])); s = R["model"].classes_[P[0].argmax()]
    e, t = emotion(text, s), topic(text); hits = [w for w in tokens(text) if w in POS | NEG]
    why = f"Classified {s.lower()} by {R['best']}" + (f"; influential words: {', '.join(hits[:4])}." if hits else ".") + f" Emotion '{e}' comes from keyword rules; category '{t}' from category keywords."
    return {"sentiment": s, "confidence": round(float(P.max()) * 100, 1), "score": round(float(score_of(R["model"], P)[0]), 2), "emotion": e, "topic": t, "explanation": why}
