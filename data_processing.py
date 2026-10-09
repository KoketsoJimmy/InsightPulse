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

VALID = ["Positive", "Neutral", "Negative"]
_DEMO = {  # topic -> (positive, neutral, negative) example sentences, all synthetic
 "Delivery": (["Delivery was fast and the courier was friendly", "My order arrived a day early"], ["The parcel arrived on Friday", "I received a tracking update this morning"], ["Delivery was extremely late", "My order arrived three days late and the box was damaged", "The courier never called and the parcel was delayed"]),
 "Customer service": (["The support agent was helpful and polite", "Customer service resolved my issue quickly"], ["I spoke to an agent about my account", "I contacted the helpdesk on Tuesday"], ["I waited two hours for the support team to reply", "The service agent was rude and unhelpful", "Nobody answered my calls to customer service"]),
 "Product quality": (["The product quality is excellent and durable", "Great item, well made and exactly as described"], ["I bought the item last week", "The product arrived in a plain box"], ["The product broke after two days", "Poor quality, the item was faulty and defective"]),
 "Pricing": (["Great value for the price", "Prices are fair and there are no hidden fees"], ["I checked the pricing page", "The fee is listed on the invoice"], ["The prices are too expensive for what you get", "Hidden fees made the cost unacceptable"]),
 "Website/app": (["The app is easy to use and fast", "The website is smooth and checkout was simple"], ["I opened the app this morning", "I installed the latest update"], ["The app keeps crashing", "The website is slow and the login fails"]),
 "Returns/refunds": (["The return process was easy and my refund came quickly", "Exchange was simple and quick"], ["I requested a return yesterday", "I submitted a refund form"], ["My refund still has not arrived after weeks", "The return process was confusing and slow"]),
}
def demo_data(n: int = 240, seed: int = 11) -> pd.DataFrame:
    """Small SYNTHETIC feedback dataset with dates and ratings (no real people)."""
    r = random.Random(seed); rows = []
    for i in range(n):
        t = r.choice(list(_DEMO)); day = i * 180 // n; pn = 0.5 if (t == "Delivery" and day > 90) else 0.3
        k = r.choices([0, 1, 2], [1 - pn - .15, .15, pn])[0]
        sent = ["Positive", "Neutral", "Negative"][k]
        rating = {"Positive": r.choice([4, 5]), "Neutral": 3, "Negative": r.choice([1, 2])}[sent]
        rows.append((i + 1, str(dt.date(2026, 1, 1) + dt.timedelta(days=day)), r.choice(_DEMO[t][k]) + r.choice(["", ".", " today.", " this week."]), rating, sent))
    return pd.DataFrame(rows, columns=["id", "date", "text", "rating", "sentiment"])

def quality_report(raw: pd.DataFrame, col: str, sent_col=None) -> dict:
    """Dynamic data-quality metrics plus a 0-100 score and traffic-light checks."""
    n = len(raw); txt = raw[col].astype(str) if col in raw else pd.Series([""] * n)
    empty_mask = (raw[col].isna() | (txt.str.strip() == "")) if col in raw else pd.Series([True] * n)
    empty, dup = int(empty_mask.sum()), int(raw.duplicated().sum())
    miss = int(raw.isna().sum().sum()); cells = max(n * len(raw.columns), 1); valid = txt[~empty_mask]
    avg = round(float(valid.str.split().str.len().mean()), 1) if len(valid) else 0.0
    inv = None
    if sent_col in raw: inv = int((raw[sent_col].isna() | ~raw[sent_col].astype(str).str.strip().str.title().isin(VALID)).sum())
    p = lambda x: x / max(n, 1) * 100
    score = 100 - min(25, miss / cells * 250) - min(20, p(dup) * 2) - min(25, p(empty) * 2.5) - (min(20, p(inv) * 2) if inv is not None else 0) - (30 if n < 30 else 15 if n < 100 else 0) - (10 if avg < 3 else 0)
    score = int(max(0, round(score))); lv = lambda bad, warn=0: "bad" if bad > 10 else "warn" if bad > warn else "good"
    checks = [(lv(p(dup)), "No duplicate records" if not dup else f"{dup} duplicate rows ({p(dup):.1f}%)"),
              (lv(miss / cells * 100, 0), "No missing values" if not miss else f"{miss} missing cells ({miss / cells * 100:.1f}% of all cells)"),
              (lv(p(empty)), "No empty feedback records" if not empty else f"{empty} empty feedback records"),
              ("good" if n >= 100 else "warn" if n >= 30 else "bad", f"{n} rows" + ("" if n >= 100 else " (small dataset: results are less reliable)")),
              ("good" if avg >= 3 else "warn", f"Average feedback length: {avg} words")]
    checks.append(("warn", "No sentiment column selected: pseudo-labels will be generated") if inv is None else (lv(p(inv)), "All sentiment labels valid" if not inv else f"{inv} records have missing or invalid sentiment labels"))
    return dict(rows=n, columns=len(raw.columns), missing=miss, duplicates=dup, empty=empty, invalid_sentiment=inv, avg_words=avg, unique_text=int(valid.nunique()), score=score, level="good" if score >= 85 else "warn" if score >= 60 else "bad", checks=checks)
