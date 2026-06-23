# 🔍 Product Review NLP Analyser

> An end-to-end NLP pipeline that automatically classifies sentiment, extracts named entities (brands & products), generates a 2-line summary, and enables semantic search — powered by BERT, BART, SpaCy, and SBERT.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![SpaCy](https://img.shields.io/badge/SpaCy-3.7-09A3D5?logo=spacy)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)
![SBERT](https://img.shields.io/badge/SBERT-sentence--transformers-green)

---

## Business Problem

E-commerce platforms receive **millions of product reviews daily**. Manual analysis is impossible at scale. Businesses need to:

- Know if customer sentiment is shifting negative for a product *(early quality warning)*
- Identify which brands and products are being discussed *(competitor intelligence)*
- Get the gist of 500 reviews without reading all 500 *(faster decisions)*
- Find all reviews about a specific issue without knowing the exact words used *(semantic discovery)*

This pipeline automates all four — turning raw review text into structured, actionable insights in seconds.

---

## Features

| # | Feature | Technology | What it does |
|---|---|---|---|
| 1 | **Sentiment Classification** | DistilBERT (SST-2) | Labels each review POSITIVE or NEGATIVE with confidence % |
| 2 | **Named Entity Recognition** | SpaCy en_core_web_sm | Extracts brands, products, prices, locations |
| 3 | **Text Summarisation** | DistilBART (CNN/DM) | Generates a 2-line abstractive summary |
| 4 | **Semantic Search** | SBERT all-MiniLM-L6-v2 | Finds reviews by meaning, not exact keywords |

---

## Architecture

```
User Input (review text / CSV / search query)
                    │
                    ▼
    ┌───────────────────────────────────────────┐
    │              nlp_pipeline.py              │
    │                                           │
    │  ┌─────────────┐                          │
    │  │  DistilBERT │ ──→ Sentiment + Score    │
    │  └─────────────┘                          │
    │  ┌─────────────┐                          │
    │  │  SpaCy NER  │ ──→ Brands / Products    │
    │  └─────────────┘                          │
    │  ┌─────────────┐                          │
    │  │ DistilBART  │ ──→ 2-line Summary       │
    │  └─────────────┘                          │
    └───────────────────────────────────────────┘
                    │
    ┌───────────────────────────────────────────┐
    │           semantic_search.py              │
    │                                           │
    │  ┌──────────────────────┐                 │
    │  │ SBERT MiniLM-L6-v2  │ ──→ Embeddings  │
    │  │ + Cosine Similarity  │ ──→ Top-K Match │
    │  └──────────────────────┘                 │
    └───────────────────────────────────────────┘
                    │
                    ▼
           Streamlit UI (app.py)
    ┌──────────┬──────────────┬─────────────────┐
    │  Single  │  Batch CSV   │ Semantic Search  │
    │  Review  │   Upload     │     Tab          │
    └──────────┴──────────────┴─────────────────┘
```

---

## Models Used

| Task | Model | Why |
|---|---|---|
| Sentiment Classification | `distilbert-base-uncased-finetuned-sst-2-english` | 40% smaller than BERT, 60% faster, 97% accuracy. Fine-tuned on Stanford Sentiment Treebank. |
| Named Entity Recognition | `SpaCy en_core_web_sm` | Lightweight statistical NER. Detects ORG, PRODUCT, MONEY, PERSON — all relevant to reviews. |
| Text Summarisation | `sshleifer/distilbart-cnn-12-6` | Distilled BART trained on CNN/DailyMail. Generates abstractive summaries, not copy-pasted sentences. |
| Semantic Search | `all-MiniLM-L6-v2` | 22MB, 384-dim embeddings, trained on 1B sentence pairs. Best speed/quality balance for CPU. |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| HuggingFace Transformers | BERT sentiment + BART summarisation |
| sentence-transformers | SBERT semantic search embeddings |
| SpaCy 3.7 | Named Entity Recognition |
| PyTorch | Model inference backend |
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
├── nlp_pipeline.py       # Core NLP — sentiment, NER, summarisation
├── semantic_search.py    # SBERT embeddings + cosine similarity search
├── app.py                # Streamlit UI — 3 tabs
├── sample_reviews.csv    # 8 sample reviews (Sony, Samsung, Apple, Bosch, Philips...)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Sample Output

### Single Review Analysis
**Input:**
> *"I bought the Sony WH-1000XM5 headphones last month and they are absolutely incredible. The noise cancellation is the best I have ever experienced. Battery life lasts the full 30 hours as advertised."*

| Field | Output |
|---|---|
| Sentiment | 🟢 POSITIVE (98.4% confidence) |
| Entities | Sony → Brand/Company, WH-1000XM5 → Product |
| Summary | Sony WH-1000XM5 headphones deliver exceptional noise cancellation and 30-hour battery life, highly recommended. |

### Semantic Search
**Query:** `"battery drains too fast"`

Finds reviews about battery issues — even ones that never use the word "battery":

| Rank | Matched Review | Similarity |
|---|---|---|
| #1 | *"Samsung Galaxy A54 battery drains within 6 hours..."* | 🟠 Moderate |
| #2 | *"OnePlus Nord CE 3 heats up significantly..."* | 🔴 Low |
| #3 | *"Philips air fryer stopped heating after 2 weeks..."* | 🔴 Low |

> Low scores are expected with only 8 sample reviews. With 1000+ reviews on the same topic, top results score 80%+.

---

## Key NLP Concepts Demonstrated

| Concept | Where |
|---|---|
| Transformer architecture + Self-Attention | DistilBERT, DistilBART |
| Bidirectional encoding | DistilBERT sentiment classifier |
| Transfer learning + Fine-tuning | All 3 HuggingFace models |
| Seq2Seq generation | DistilBART summarisation |
| Named Entity Recognition | SpaCy NER pipeline |
| Sentence embeddings + Mean pooling | SBERT all-MiniLM-L6-v2 |
| Cosine similarity | Semantic search scoring |
| Dense retrieval | Foundation of semantic search tab |

---

## Real-World Applications

- **E-commerce** — auto-flag products with declining sentiment before brand damage occurs
- **FMCG / Electronics brands** — find all complaints about a specific issue (battery, heating, delivery) without manual keyword matching
- **Market research** — voice-of-customer reports from scraped review data at scale
- **Customer support** — summarise a customer's full review history so agents resolve issues faster
- **RAG pipelines** — the semantic search module is a direct building block for Retrieval Augmented Generation systems

---

## Author

**Divya Palanikumar**
Analytics Engineer | AIML Practitioner
[GitHub](https://github.com/Divya-0709) · [Email](mailto:palanikumardivya92@gmail.com)
