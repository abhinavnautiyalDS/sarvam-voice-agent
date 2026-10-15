from __future__ import annotations
import io, re, time, unicodedata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

LABELS = ("english", "hinglish")
OUTPUT_LABELS = ("english", "hinglish", "other", "uncertain")

# Common Romanized Hindi cues. This is an explicit heuristic layer for short
# turns; expand it using reviewed call examples rather than auto-labeling them.
ROMANIZED_HINDI_CUES = set("""
hai hain ho hun hoga hogi honge tha thi mujhe mera meri mere aap apko aapko
aapka aapki tum tumhe tu kya kaise kaisa kaunsi kaun kab kahan kaha kyun kyu kyon
nahi nahin na nhi haan han ji jii achha acha accha thik theek kr do karo karna
chahiye chahte chahta bata batao bataiye bataye bol raha rahi abhi kal aaj par
se ko ka ki ke mein mai mey main hum ham hamara tumhara yeh ye woh vo hai
hain namaste dhanyawad shukriya yahan wahan kitna kitne kitni please batao
""".split())
SINGLE_WORD_UNCERTAIN = {"hello", "hi", "hey", "bye", "goodbye", "thanks", "okay", "ok", "yes", "no", "sure", "please", "right", "correct", "great", "fine"}
DEVANAGARI_RANGE = re.compile(r"[\\u0900-\\u097F]")
NON_LATIN_ALPHA = re.compile(r"[^\\W\\d_]", re.UNICODE)


def script_profile(text: str) -> tuple[bool, bool, bool]:
    """Return (has_devanagari, has_other_non_latin, has_latin_letters)."""
    has_devanagari = has_other_non_latin = has_latin = False
    for ch in str(text or ""):
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            has_latin = True
        elif "DEVANAGARI" in name:
            has_devanagari = True
        else:
            has_other_non_latin = True
    return has_devanagari, has_other_non_latin, has_latin


def romanized_hindi_cue(text: str) -> bool:
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())
    # A single cue can be enough for a short mixed utterance ("haan please").
    return any(token in ROMANIZED_HINDI_CUES for token in tokens)


class OtherLanguageRejector:
    """English/Hinglish model with deterministic script/cue rules and abstention.

    Policy: Devanagari (pure Hindi or mixed script) => Hinglish; other non-Latin
    scripts without Devanagari => OTHER; Romanized Hindi cues => Hinglish;
    one-word English greetings/acknowledgements => UNCERTAIN; low-margin Latin
    text => UNCERTAIN. Remaining Latin text uses the trained binary classifier.
    """
    def __init__(self, model, min_margin: float = 0.35):
        self.model = model
        self.min_margin = float(min_margin)
        self.classes_ = np.array(OUTPUT_LABELS, dtype=object)

    def _route(self, text):
        cleaned = str(text or "").strip()
        has_devanagari, has_other_non_latin, has_latin = script_profile(cleaned)
        if not cleaned:
            return "uncertain", None
        if has_devanagari:
            return "hinglish", None
        if has_other_non_latin and not has_latin:
            return "other", None
        if has_other_non_latin and has_latin:
            return "other", None
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", cleaned.lower())
        if cleaned.lower() in {"okay sir", "ok sir", "yes sir", "no sir"}:
            return "uncertain", None
        if len(tokens) == 1 and tokens[0] in SINGLE_WORD_UNCERTAIN:
            return "uncertain", None
        if romanized_hindi_cue(cleaned):
            return "hinglish", None
        if not has_latin:
            return "uncertain", None
        if len(tokens) < 2 or sum(ch.isalpha() for ch in cleaned) < 5:
            return "uncertain", None
        score = float(np.asarray(self.model.decision_function([cleaned])).reshape(-1)[0])
        if abs(score) < self.min_margin:
            return "uncertain", score
        return str(self.model.classes_[1] if score > 0 else self.model.classes_[0]), score

    def predict(self, texts):
        return np.asarray([self._route(text)[0] for text in texts], dtype=object)

    def predict_proba(self, texts):
        rows = []
        for text in texts:
            label, score = self._route(text)
            if label == "hinglish" and score is None:
                rows.append([0.0, 1.0, 0.0, 0.0])
            elif label == "other":
                rows.append([0.0, 0.0, 1.0, 0.0])
            elif label == "uncertain":
                rows.append([0.0, 0.0, 0.0, 1.0])
            else:
                strength = 1.0 / (1.0 + np.exp(-min(abs(float(score)), 30.0)))
                rows.append([1.0-strength, strength, 0.0, 0.0] if score > 0 else [strength, 1.0-strength, 0.0, 0.0])
        return np.asarray(rows, dtype=float)


def with_other_rejection(model, min_margin: float = 0.35):
    return OtherLanguageRejector(model, min_margin=min_margin)



def normalize_binary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a text/label frame and retain only binary training labels."""
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Dataset must contain text and label columns.")
    out = frame[["text", "label"]].copy()
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    out["label"] = out["label"].fillna("").astype(str).str.lower().str.strip()
    out = out[out["label"].isin(LABELS) & out["text"].ne("")]
    out = out.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    return out


def binary_corpus(path) -> pd.DataFrame:
    """Load a repository dataset and keep only English/Hinglish rows."""
    frame = pd.read_csv(path)
    return normalize_binary_frame(frame)


def _candidate_models():
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2,
        max_features=180000, sublinear_tf=True, strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        max_features=220000, sublinear_tf=True,
    )
    features = FeatureUnion([("word", word), ("char", char)])
    return {
        "LinearSVC": Pipeline([("tfidf", features), ("clf", LinearSVC(C=1.0, class_weight="balanced"))]),
        "LogisticRegression": Pipeline([("tfidf", features), ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", C=4.0))]),
        "SGDClassifier": Pipeline([("tfidf", features), ("clf", SGDClassifier(loss="log_loss", max_iter=1500, tol=1e-3, class_weight="balanced", random_state=42))]),
        "ComplementNB": Pipeline([("tfidf", features), ("clf", ComplementNB(alpha=0.2))]),
    }


def evaluate_candidates(train: pd.DataFrame, test: pd.DataFrame):
    """Fit candidate pipelines and evaluate on the fixed held-out corpus split."""
    train = normalize_binary_frame(train)
    test = normalize_binary_frame(test)
    if train.empty or test.empty or train["label"].nunique() != 2 or test["label"].nunique() != 2:
        raise ValueError("Training and test data must each contain both english and hinglish labels.")
    X_train, y_train = train["text"].tolist(), train["label"].tolist()
    X_test, y_test = test["text"].tolist(), test["label"].tolist()
    results, models = {}, {}
    for name, model in _candidate_models().items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - started
        pred = model.predict(X_test)
        latencies = []
        for sample in X_test[: min(500, len(X_test))]:
            t0 = time.perf_counter()
            model.predict([sample])
            latencies.append((time.perf_counter() - t0) * 1000)
        decision = model.predict(X_train)
        report = classification_report(y_test, pred, labels=list(LABELS), output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, pred, labels=list(LABELS)).tolist()
        try:
            import pickle
            model_size_mb = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / (1024 * 1024)
        except Exception:
            model_size_mb = 0.0
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, labels=list(LABELS), average="macro", zero_division=0)),
            "classification_report": report,
            "confusion_matrix": matrix,
            "latency": {
                "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
                "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            },
            "train_seconds": float(train_seconds),
            "model_size_mb": float(model_size_mb),
            "generalization_gap_accuracy": float(accuracy_score(y_train, decision) - accuracy_score(y_test, pred)),
        }
        models[name] = model
    return results, models
