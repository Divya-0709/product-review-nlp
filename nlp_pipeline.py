"""
nlp_pipeline.py
Core NLP logic for the Product Review Analyser.

Three tasks:
  1. Sentiment Classification  → DistilBERT (fast, accurate)
  2. Named Entity Recognition  → SpaCy en_core_web_sm
  3. Text Summarisation        → DistilBART (lightweight BART)
"""

from transformers import pipeline as hf_pipeline
import spacy

# ── Model loading (runs once at import time) ─────────────────────────────────

print("Loading sentiment model...")
_sentiment_model = hf_pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

print("Loading summarisation model...")
_summariser = hf_pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
)

print("Loading SpaCy NER model...")
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "SpaCy model not found. Run:  python -m spacy download en_core_web_sm"
    )

print("All models loaded.")

# ── Entity labels we care about for product reviews ──────────────────────────
RELEVANT_LABELS = {
    "ORG":     "Brand / Company",
    "PRODUCT": "Product",
    "PERSON":  "Person",
    "GPE":     "Location",
    "MONEY":   "Price",
}


# ── Individual task functions ─────────────────────────────────────────────────

def classify_sentiment(text: str) -> dict:
    """
    Returns sentiment label (POSITIVE / NEGATIVE) and confidence score.
    Truncates to 512 chars to stay within model token limit.
    """
    result = _sentiment_model(text[:512])[0]
    return {
        "label": result["label"],
        "confidence": round(result["score"] * 100, 1),
    }


def extract_entities(text: str) -> list:
    """
    Runs SpaCy NER and returns entities relevant to product reviews.
    Deduplicates by (text, label) pair.
    """
    doc = _nlp(text)
    seen = set()
    entities = []

    for ent in doc.ents:
        if ent.label_ in RELEVANT_LABELS:
            key = (ent.text.strip(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({
                    "entity":      ent.text.strip(),
                    "type":        RELEVANT_LABELS[ent.label_],
                    "spacy_label": ent.label_,
                })

    return entities


def summarise_review(text: str) -> str:
    """
    Returns a 1-2 sentence summary.
    Falls back to the original text if it is too short to summarise.
    """
    word_count = len(text.split())

    if word_count < 25:
        return text  # already short — no summarisation needed

    result = _summariser(
        text[:900],
        max_length=60,
        min_length=20,
        do_sample=False,
    )
    return result[0]["summary_text"]


# ── Main entry point ──────────────────────────────────────────────────────────

def analyse_review(text: str) -> dict:
    """
    Runs the full pipeline on a single review.

    Returns:
        {
            "sentiment":   { "label": "POSITIVE", "confidence": 97.3 },
            "entities":    [ { "entity": "Sony", "type": "Brand / Company" } ],
            "summary":     "The Sony headphones offer excellent sound quality.",
            "input_text":  <original review>
        }
    """
    text = text.strip()
    if not text:
        raise ValueError("Review text cannot be empty.")

    return {
        "input_text": text,
        "sentiment":  classify_sentiment(text),
        "entities":   extract_entities(text),
        "summary":    summarise_review(text),
    }
