"""
Streamlit web app for fake news detection.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import streamlit as st

from utilis import (
    clean_text,
    explain,
    extract_article,
    fact_check_lookup,
    fake_probability,
    verdict,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "fake_news_model.joblib"
METRICS_PATH = PROJECT_DIR / "model" / "metrics.json"
CONFUSION_PATH = PROJECT_DIR / "model" / "confusion_matrix.png"
TOP_WORDS_PATH = PROJECT_DIR / "model" / "top_words.png"

MIN_BODY_CHARACTERS = 100  # very short text gives unreliable scores

st.set_page_config(page_title="Fake News Detector", page_icon="📰")


@st.cache_resource
def load_model():
    """Load the trained model once and keep it in memory."""
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Load the saved evaluation numbers, if train_model.py has produced them."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


metrics = load_metrics()

# ----- Sidebar: how it works and how well it performed -----
with st.sidebar:
    st.header("About the model")
    st.write(
        "The app cleans the text, turns words into TF-IDF scores, and uses a linear "
        "model to estimate how likely the article is to be fake."
    )
    if metrics:
        chosen = metrics["selected_model"]
        test = metrics["test"][chosen]
        st.write(f"Deployed model: **{chosen}**")
        st.metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
        st.metric("Test F1 (fake class)", f"{test['f1']:.3f}")
        st.caption(f"Trained on: {metrics['dataset']} ({metrics['n_train']} training articles)")
        for warning in metrics["warnings"]:
            st.warning(warning)
    st.caption("The score is a statistical guess based on writing patterns. It is not a fact-check.")

# ----- Main page -----
st.title("📰 Fake News Detector")
st.write("Paste an article, or give a link, to check whether it looks like real or fake news.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train_model.py` first, then refresh this page.")
    st.stop()

model = load_model()

url = st.text_input("Article URL (optional)", placeholder="https://...")
title = st.text_input("Headline (optional)", placeholder="Paste the headline here")
body = st.text_area("Article text (optional if you gave a URL)", height=250,
                    placeholder="Paste the article body here...")

if st.button("Check article", type="primary"):
    article_title = title.strip()
    article_body = body.strip()
    error = None

    if url.strip():
        try:
            fetched_title, fetched_text = extract_article(url.strip())
            # Pasted text takes priority; the URL only fills in what is missing
            article_title = article_title or fetched_title
            article_body = article_body or fetched_text
            st.info(f"Read {len(fetched_text)} characters from the link.")
        except Exception as exc:  # network errors and "no article found"
            error = str(exc) or "Could not read that link. Paste the article text instead."

    if error:
        st.error(error)
    elif len(article_body) < MIN_BODY_CHARACTERS:
        st.warning(f"Please provide at least {MIN_BODY_CHARACTERS} characters of article text.")
    elif not clean_text(f"{article_title} {article_body}"):
        st.warning("Nothing useful was left after cleaning the text. Check what you pasted.")
    else:
        combined = f"{article_title} {article_body}".strip()
        prob_fake = fake_probability(model, combined)
        result = verdict(prob_fake)

        if result == "fake":
            st.error(f"Likely FAKE news (fake score {prob_fake:.0%})")
        elif result == "real":
            st.success(f"Likely REAL news (fake score {prob_fake:.0%})")
        else:
            st.warning(
                f"UNCERTAIN (fake score {prob_fake:.0%}). The model cannot tell. "
                "Verify this article with trusted sources."
            )
        st.progress(prob_fake)

        # Show which words pushed the score, so the result can be checked
        fake_terms, real_terms = explain(model, combined)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Words pushing toward FAKE**")
            st.write(", ".join(term for term, _ in fake_terms) or "none found")
        with col2:
            st.write("**Words pushing toward REAL**")
            st.write(", ".join(term for term, _ in real_terms) or "none found")

        # ----- Fact-check lookup -----
        st.subheader("Published fact-checks")
        query = article_title or article_body.split(".")[0]
        try:
            checks = fact_check_lookup(query)
        except Exception as exc:
            checks = "error"
            st.warning(f"Fact-check lookup failed: {exc}")

        if checks is None:
            st.info(
                "Fact-check lookup is off. Set the GOOGLE_FACT_CHECK_API_KEY environment "
                "variable to turn it on."
            )
        elif checks == "error":
            pass  # warning already shown above
        elif not checks:
            st.write("No published fact-checks matched this headline.")
        else:
            for check in checks:
                line = f"- **{check['rating']}** ({check['publisher']}): {check['claim']}"
                if check["url"]:
                    line += f" [source]({check['url']})"
                st.write(line)

        st.caption(
            "A score is a statistical signal, not proof. "
            "Check important news against trusted sources before sharing it."
        )

# ----- Training results -----
if metrics and CONFUSION_PATH.exists() and TOP_WORDS_PATH.exists():
    with st.expander("How the model was evaluated"):
        st.image(CONFUSION_PATH, caption="Where the model makes mistakes (test set)")
        st.image(TOP_WORDS_PATH, caption="Words the model relies on most")
