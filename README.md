# 🔍 Product Review NLP Analyser

> An end-to-end NLP pipeline that automatically classifies sentiment, extracts named entities (brands & products), and generates a 2-line summary from product reviews — powered by BERT, BART, and SpaCy.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![SpaCy](https://img.shields.io/badge/SpaCy-3.7-09A3D5?logo=spacy)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)

---

## Business Problem

E-commerce platforms receive **millions of product reviews daily**. Manual analysis is impossible at scale. Businesses need to know:

- Is customer sentiment shifting negative for a product? *(early warning for quality issues)*
- Which brands and products are being talked about? *(competitor intelligence)*
- What is the gist of 500 reviews without reading all 500? *(faster decisions)*

This pipeline automates all three — turning raw review text into structured, actionable insights in seconds.

---

## Demo

### Single Review Analysis
Paste any product review → get sentiment, entities, and a summary instantly.

### Batch CSV Analysis
Upload a CSV of reviews → pipeline processes all rows → download enriched results with sentiment labels, confidence scores, extracted entities, and summaries.

---

## Architecture

```
User Input (review text or CSV)
           │
           ▼
    ┌──────────────────────────────────────┐
    │         nlp_pipeline.py              │
    │                                      │
    │  ┌─────────────┐                     │
    │  │  DistilBERT │ → Sentiment Label   │
    │  │  (BERT SST-2│   + Confidence %    │
    │  └─────────────┘                     │
    │                                      │
    │  ┌─────────────┐                     │
    │  │    SpaCy    │ → Brand / Product / │
    │  │ en_core_web │   Person / Price    │
    │  └─────────────┘                     │
    │                                      │
    │  ┌─────────────┐                     │
    │  │ DistilBART  │ → 2-line Summary    │
    │  │  (CNN/DM)   │                     │
    │  └─────────────┘                     │
    └──────────────────────────────────────┘
           │
           ▼
    Streamlit UI  (app.py)
    Single review view  |  Batch CSV + download
```

---

## Models Used

| Task | Model | Why This Model |
|---|---|---|
| Sentiment Classification | `distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT is 40% smaller and 60% faster than BERT, retaining 97% accuracy. Fine-tuned on SST-2 (Stanford Sentiment Treebank). |
| Named Entity Recognition | `SpaCy en_core_web_sm` | Lightweight, fast NER model. Detects ORG (brands), PRODUCT, PERSON, MONEY — all relevant to product reviews. |
| Text Summarisation | `sshleifer/distilbart-cnn-12-6` | Distilled BART trained on CNN/DailyMail. Abstractive summarisation — generates human-like summaries, not just extracted sentences. |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| HuggingFace Transformers | BERT sentiment + BART summarisation |
| SpaCy 3.7 | Named Entity Recognition |
| Streamlit | Interactive web UI |
| Pandas | Batch CSV processing |

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Divya-0709/product-review-nlp.git
cd product-review-nlp

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Install SpaCy language model
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# 4. Run the app
streamlit run app.py
```

> **Note:** First run downloads ~500MB of HuggingFace models. They are cached locally after that.

---

## Project Structure

```
product-review-nlp/
├── nlp_pipeline.py      # Core NLP logic — sentiment, NER, summarisation
├── app.py               # Streamlit UI — single review + batch CSV
├── sample_reviews.csv   # 8 sample reviews across Sony, Samsung, Apple, Bosch etc.
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Sample Output

**Input review:**
> "I bought the Sony WH-1000XM5 headphones last month and they are absolutely incredible. The noise cancellation is the best I have ever experienced. Battery life lasts the full 30 hours as advertised."

| Field | Output |
|---|---|
| Sentiment | 🟢 POSITIVE (98.4% confidence) |
| Entities | Sony → Brand/Company, WH-1000XM5 → Product |
| Summary | Sony WH-1000XM5 headphones offer exceptional noise cancellation and 30-hour battery life, delivering on all advertised claims. |

---

## Key NLP Concepts Demonstrated

- **Transformer architecture** — BERT's bidirectional encoder for contextual text understanding
- **Transfer learning & fine-tuning** — using pretrained models adapted for specific downstream tasks
- **Seq2Seq generation** — BART's encoder-decoder architecture for abstractive summarisation
- **Named Entity Recognition** — SpaCy's statistical NER pipeline with entity deduplication
- **NLP pipeline design** — modular, task-specific functions composed into a single `analyse_review()` call
- **Batch inference** — processing multiple reviews with progress tracking and CSV export

---

## Real-World Applications

- **E-commerce platforms** — auto-flag products with declining sentiment before they damage brand reputation
- **FMCG / electronics brands** — track which product features draw praise or complaints at scale
- **Market research** — generate voice-of-customer reports from scraped review data
- **Customer support** — summarise a customer's review history so agents resolve issues faster

---

## Author

**Divya Palanikumar**  
Analytics Engineer | AIML Practitioner  
[GitHub](https://github.com/Divya-0709) · [Email](mailto:palanikumardivya92@gmail.com)
