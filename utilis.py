"""
Shared code used by training, the web app, the API, and the tests.

Design rule: text cleaning is part of the model pipeline (it is the
TfidfVectorizer's preprocessor), so model.predict(raw_text) works directly.
Training and prediction can never disagree about how text is cleaned.
"""

import os
import re

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Scores between these two values are shown as UNCERTAIN
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65

FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FACT_CHECK_KEY_ENV = "GOOGLE_FACT_CHECK_API_KEY"

# Phrases that tell the model WHERE an article came from, not whether it is true.
# Example: "WASHINGTON (Reuters) -" appears in many real articles in the Kaggle data
# and in almost no fake ones, so the model would learn the source instead of the content.
SOURCE_PATTERNS_CASE_SENSITIVE = [
    r"\b[A-Z][A-Z .,'\-]{1,50}\(Reuters\)\s*-?",
    r"\(Reuters\)",
]
SOURCE_PATTERNS_LOWERCASE = [
    r"featured image via [^.\n]{0,60}",
    r"image via \S+",
    r"pic\.twitter\.com/\S+",
    r"\(photo[^)]*\)",
]


def clean_text(text):
    """
    Prepare raw article text for the TF-IDF vectorizer.

    Steps:
      1. Remove wire-service and photo-credit phrases (source clues, not content)
      2. Lowercase
      3. Remove URLs and HTML tags
      4. Keep letters only (removes digits and punctuation)
      5. Collapse extra spaces

    Stop words are KEPT on purpose. Words like "not", "no" and "never" carry
    meaning in news ("not true", "no evidence"), and removing them hurts the model.
    """
    if not isinstance(text, str):
        return ""

    for pattern in SOURCE_PATTERNS_CASE_SENSITIVE:
        text = re.sub(pattern, " ", text)

    text = text.lower()

    for pattern in SOURCE_PATTERNS_LOWERCASE:
        text = re.sub(pattern, " ", text)

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def style_features(texts):
    """
    Simple writing-style numbers computed from the RAW text.

    Fake news often uses more capital letters, exclamation marks and question
    marks. Cleaning removes these clues, so we measure them here before cleaning.
    """
    rows = []
    for text in texts:
        text = text if isinstance(text, str) else ""
        letters = [c for c in text if c.isalpha()]
        upper_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
        exclamations_per_100_chars = 100 * text.count("!") / max(len(text), 1)
        words = text.split()
        avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0
        rows.append([
            upper_ratio,
            exclamations_per_100_chars,
            text.count("?"),
            np.log1p(len(words)),
            avg_word_length,
        ])
    return np.array(rows, dtype=float)


def build_pipeline(classifier=None, use_style=True):
    """
    Build the full model: raw text in, fake/real prediction out.

    Feature groups joined together:
      - word features: TF-IDF on cleaned text (single words and word pairs)
      - style features (optional): capital letters, exclamation marks, etc.

    use_style=False is needed for Multinomial Naive Bayes, which only accepts
    non-negative numbers, and the scaled style features can be negative.
    """
    if classifier is None:
        classifier = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")

    transformers = [
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,  # cleaning happens inside the model
            ngram_range=(1, 2),  # "secret" and "secret cure"
            min_df=2,  # ignore words that appear in only one article
            max_df=0.9,  # ignore words in more than 90% of articles
            max_features=20000,
            sublinear_tf=True,
        )),
    ]
    if use_style:
        transformers.append(("style", Pipeline([
            ("extract", FunctionTransformer(style_features)),
            ("scale", StandardScaler()),
        ])))

    return Pipeline([
        ("features", FeatureUnion(transformers)),
        ("clf", classifier),
    ])


def fake_probability(model, text):
    """
    Return a fake score between 0 and 1 for one piece of raw text.

    Logistic regression gives probabilities directly. Linear SVM gives a distance
    from the decision line, which we squash into 0-1 with a sigmoid curve.
    """
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return float(model.predict_proba([text])[0][1])
    score = model.decision_function([text])[0]
    return float(1 / (1 + np.exp(-score)))


def verdict(prob_fake):
    """Turn a fake score into 'fake', 'real' or 'uncertain'."""
    if prob_fake >= UNCERTAIN_HIGH:
        return "fake"
    if prob_fake <= UNCERTAIN_LOW:
        return "real"
    return "uncertain"


def top_terms(model, n=20):
    """
    The words with the strongest weights in a fitted linear model.

    Returns two lists of (word, weight): words pushing toward FAKE and toward REAL.
    Only the word features are included, not the style features.
    """
    tfidf = dict(model.named_steps["features"].transformer_list)["tfidf"]
    clf = model.named_steps["clf"]
    names = tfidf.get_feature_names_out()
    weights = clf.coef_[0][: len(names)]

    order = np.argsort(weights)
    fake = [(names[i], float(weights[i])) for i in order[::-1][:n]]
    real = [(names[i], float(weights[i])) for i in order[:n]]
    return fake, real


def explain(model, text, n=5):
    """
    Which words in THIS article pushed the score toward fake or real.

    For each word in the article: (how strongly it appears) x (its weight in the model).
    Returns two lists of (word, contribution): fake-pushing and real-pushing words.
    """
    if "features" in model.named_steps:
        tfidf = dict(model.named_steps["features"].transformer_list)["tfidf"]
    else:
        tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = tfidf.get_feature_names_out()

    vector = tfidf.transform([text]).toarray()[0]
    contributions = vector * clf.coef_[0][: len(names)]
    order = np.argsort(contributions)

    fake = [(names[i], float(contributions[i]))
            for i in order[::-1][:n] if contributions[i] > 0]
    real = [(names[i], float(-contributions[i]))
            for i in order[:n] if contributions[i] < 0]
    return fake, real


def extract_article(url):
    """
    Download a web page and return (title, main article text).

    Uses trafilatura, which removes menus, ads and comments and keeps the article body.
    Raises ValueError if the page cannot be downloaded or has no article text.
    """
    import trafilatura  # imported here so the rest of the project works without it

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Could not download that page. Paste the article text instead.")

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError("No article text was found on that page. Paste the text instead.")

    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else ""
    return title, text


def fact_check_lookup(query, max_results=3):
    """
    Search published fact-checks with the Google Fact Check Tools API.

    Returns:
      None           if no API key is set (the feature is off)
      []             if the API found no matching fact-checks
      list of dicts  with keys: claim, rating, publisher, url

    Raises requests.RequestException if the network call fails.
    Get a key from Google Cloud (enable "Fact Check Tools API") and set it as
    the environment variable GOOGLE_FACT_CHECK_API_KEY.
    """
    api_key = os.environ.get(FACT_CHECK_KEY_ENV)
    if not api_key:
        return None

    response = requests.get(
        FACT_CHECK_URL,
        params={"query": query[:200], "key": api_key, "pageSize": max_results},
        timeout=10,
    )
    response.raise_for_status()

    results = []
    for claim in response.json().get("claims", []):
        for review in claim.get("claimReview", []):
            results.append({
                "claim": claim.get("text", ""),
                "rating": review.get("textualRating", ""),
                "publisher": review.get("publisher", {}).get("name", ""),
                "url": review.get("url", ""),
            })
    return results[:max_results]
