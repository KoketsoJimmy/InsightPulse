"""Loading, cleaning and NLTK text preprocessing."""
import re, random, datetime as dt
import pandas as pd
try:
    import nltk
    from nltk.tokenize import wordpunct_tokenize  # rule-based, needs no downloads
    try: nltk.data.find("corpora/stopwords")
    except LookupError: nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))
except Exception:  # offline fallback keeps the app working
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP
    wordpunct_tokenize = lambda t: re.findall(r"\w+|[^\w\s]", t)
STOP = set(STOP) - {"not", "no", "nor", "very"}  # negations carry sentiment
EXTRA = {"today", "week", "again", "honestly", "overall", "yesterday", "morning", "tuesday", "friday", "last"}

def clean(t):
    """Lowercase, strip URLs and special characters, normalise whitespace."""
    t = re.sub(r"https?://\S+|www\.\S+", " ", str(t).lower())
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def tokens(t):
    """Tokenise and remove stop words."""
    return [w for w in wordpunct_tokenize(clean(t)) if w not in STOP and w not in EXTRA and len(w) > 2 and w.isalnum()]

def sample_data(n=520, seed=7):
    r = random.Random(seed)
    P = {"Customer Service": ["The service was excellent and the staff were helpful", "Reception was friendly and quick", "The agent resolved my issue politely"],
         "Technology": ["The application is easy to use", "The website is fast and smooth", "Login works perfectly every time"],
         "Delivery": ["My parcel arrived early and well packed", "Delivery was fast and the courier was friendly"],
         "Pricing": ["Great value for the price", "The fees are fair and clear"],
         "Quality": ["The quality is excellent and very durable", "Well-made product, I love it"]}
    N = {"Customer Service": ["I waited for three hours before someone helped me", "The service was rude and unhelpful", "The queue was terrible and nobody cared"],
         "Technology": ["The application keeps crashing", "The website is slow and the login fails", "The app is confusing and full of bugs"],
         "Delivery": ["My parcel arrived late and damaged", "Delivery was slow and the courier never called"],
         "Pricing": ["The prices are too expensive for what you get", "Hidden fees made the cost unacceptable"],
         "Quality": ["The product broke after two days, poor quality", "Faulty item, very disappointing"]}
    U = {"Customer Service": ["I visited the branch on Tuesday", "I spoke to someone about my account"],
         "Technology": ["I installed the update yesterday", "I opened the application this morning"],
         "Delivery": ["The parcel came on Friday", "I received a delivery notification"],
         "Pricing": ["I checked the pricing page", "The fee is listed on the invoice"],
         "Quality": ["I bought the item last week", "The product arrived in a box"]}
    ext = ["", " today", " this week", " again", ", honestly", " overall"]
    rows, d0 = [], dt.date(2026, 1, 1)
    for i in range(n):
        c = r.choice(list(P)); day = i * 180 // n; late = day > 90
        pn = 0.55 if (c == "Technology" and late) else 0.30
        s = r.choices(["Positive", "Negative", "Neutral"], [1 - pn - .15, pn, .15])[0]
        base = {"Positive": P, "Negative": N, "Neutral": U}[s][c]
        rows.append((i + 1, r.choice(base) + r.choice(ext) + ".", c, str(d0 + dt.timedelta(days=day)), s))
    df = pd.DataFrame(rows, columns=["id", "text", "category", "date", "sentiment"])
    return df.drop_duplicates("text").assign() if False else df  # duplicates allowed: realistic repeats

def detect_text_col(df):
    for n in ["text", "feedback", "comment", "review", "message", "response"]:
        for c in df.columns:
            if c.lower() == n: return c
    obj = [c for c in df.columns if df[c].dtype == object]
    if not obj: return None
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())


def detect_text_col(df):
    for n in ["text", "feedback", "comment", "review", "message", "response"]:
        for c in df.columns:
            if str(c).lower() == n: return c
    obj = [c for c in df.columns if df[c].dtype == object]
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean()) if obj else None

def data_summary(df, col):
    return {"records": len(df), "columns": len(df.columns), "missing": int(df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()), "duplicates": int(df.duplicated().sum())}

def load_clean(df, col):
    """Drop missing text and duplicates; keep the original frame untouched."""
    out = df.dropna(subset=[col]); out = out[out[col].astype(str).str.strip() != ""].drop_duplicates().reset_index(drop=True)
    if out.empty: raise ValueError("The uploaded dataset is empty after cleaning.")
    out["text"] = out[col].astype(str).str.strip().str.slice(0, 1000); out["clean_text"] = out["text"].map(clean)
    return out
