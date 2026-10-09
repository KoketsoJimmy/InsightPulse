"""Train and compare lightweight scikit-learn text classifiers on one shared split."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.data_processing import clean

def compare_models(texts, labels, seed: int = 42) -> dict:
    X = [clean(t) for t in texts]
    xtr, xte, ytr, yte = train_test_split(X, labels, test_size=.25, random_state=seed, stratify=labels)
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(xtr); A, B = vec.transform(xtr), vec.transform(xte)
    cands = {"Naive Bayes": MultinomialNB(), "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
             "Random Forest": RandomForestClassifier(n_estimators=150, random_state=seed), "Linear SVM": CalibratedClassifierCV(LinearSVC(), cv=3)}
    res, models, notes = {}, {}, {}
    for name, m in cands.items():
        try:
            m.fit(A, ytr); p = m.predict(B); pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="weighted", zero_division=0)
            res[name] = {k: round(float(v) * 100, 1) for k, v in dict(Accuracy=accuracy_score(yte, p), Precision=pr, Recall=rc, F1=f1).items()}; models[name] = m
        except Exception as e: notes[name] = f"Not trained: {e}"
    if not res: raise ValueError("Model training failed: the dataset is too small or unsuitable.")
    best = max(res,  # ties keep the earliest (simplest) model
                key=lambda k: res[k]["F1"]); labs = sorted(set(labels))
    return dict(vec=vec, models=models, best=best, metrics=res, notes=notes, ntrain=len(xtr), ntest=len(xte), cm=confusion_matrix(yte, models[best].predict(B), labels=labs), cm_labels=labs)
