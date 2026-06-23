"""
app.py
Streamlit UI for the Product Review NLP Analyser.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from nlp_pipeline import analyse_review

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Review Analyser",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Product Review NLP Analyser")
st.markdown(
    "Paste a product review below. The pipeline will classify **sentiment**, "
    "extract **named entities** (brands, products), and generate a **2-line summary**."
)

# ── Tabs: Single review vs Batch CSV ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Single Review", "Batch CSV Upload", "Semantic Search"])


# ── Helper: display results ───────────────────────────────────────────────────
def display_results(result: dict):
    sentiment = result["sentiment"]
    entities  = result["entities"]
    summary   = result["summary"]

    col1, col2 = st.columns(2)

    # Sentiment
    with col1:
        st.subheader("Sentiment")
        colour = "🟢" if sentiment["label"] == "POSITIVE" else "🔴"
        st.metric(
            label=f"{colour} {sentiment['label']}",
            value=f"{sentiment['confidence']}% confidence",
        )

    # Summary
    with col2:
        st.subheader("Summary")
        st.info(summary)

    # Entities
    st.subheader("Named Entities")
    if entities:
        df = pd.DataFrame(entities)[["entity", "type"]]
        df.columns = ["Entity", "Type"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.write("No brand/product entities detected.")


# ── Tab 1: Single review ──────────────────────────────────────────────────────
with tab1:
    sample_reviews = {
        "Select a sample...": "",
        "Positive – Sony headphones": (
            "I bought the Sony WH-1000XM5 headphones last month and they are absolutely "
            "incredible. The noise cancellation is the best I have ever experienced — "
            "completely blocks out my noisy office. Battery life lasts the full 30 hours "
            "as advertised. The sound quality is rich and detailed. Highly recommend to "
            "anyone looking for premium wireless headphones. Worth every rupee."
        ),
        "Negative – Samsung phone": (
            "Very disappointed with the Samsung Galaxy A54. The battery drains within "
            "6 hours of normal usage, which is unacceptable for a phone at this price. "
            "The camera is mediocre at best despite Samsung's claims. Customer support "
            "was unhelpful when I raised the issue. I expected much better quality from "
            "a brand like Samsung. Would not recommend this phone to anyone."
        ),
        "Mixed – Amazon delivery": (
            "The Apple AirPods Pro sound quality is fantastic and the fit is comfortable. "
            "However, Amazon delivered a damaged box and one earbud had a scratch on it. "
            "The product itself from Apple is great but the packaging and delivery "
            "experience by Amazon was really poor. Had to contact customer service twice."
        ),
    }

    selected = st.selectbox("Try a sample review:", list(sample_reviews.keys()))
    default_text = sample_reviews[selected]

    review_text = st.text_area(
        "Or type / paste your own review:",
        value=default_text,
        height=180,
        placeholder="e.g. I bought the Sony WH-1000XM5 headphones and they are incredible...",
    )

    if st.button("Analyse Review", type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review to analyse.")
        else:
            with st.spinner("Running NLP pipeline..."):
                result = analyse_review(review_text)
            display_results(result)


# ── Tab 2: Batch CSV upload ───────────────────────────────────────────────────
with tab2:
    st.markdown(
        "Upload a CSV file with a column named **`review`**. "
        "The pipeline will analyse all rows and let you download the results."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_input = pd.read_csv(uploaded)

        if "review" not in df_input.columns:
            st.error("CSV must have a column named 'review'.")
        else:
            st.write(f"Found **{len(df_input)} reviews**. Preview:")
            st.dataframe(df_input.head(3), use_container_width=True)

            if st.button("Run Batch Analysis", type="primary"):
                results = []
                progress = st.progress(0)

                for i, row in df_input.iterrows():
                    try:
                        r = analyse_review(str(row["review"]))
                        results.append({
                            "review":     row["review"],
                            "sentiment":  r["sentiment"]["label"],
                            "confidence": r["sentiment"]["confidence"],
                            "entities":   ", ".join(e["entity"] for e in r["entities"]),
                            "summary":    r["summary"],
                        })
                    except Exception as e:
                        results.append({
                            "review": row["review"],
                            "sentiment": "ERROR", "confidence": 0,
                            "entities": "", "summary": str(e),
                        })
                    progress.progress((i + 1) / len(df_input))

                df_out = pd.DataFrame(results)
                st.success("Analysis complete!")
                st.dataframe(df_out, use_container_width=True)

                csv_bytes = df_out.to_csv(index=False).encode()
                st.download_button(
                    "Download Results CSV",
                    data=csv_bytes,
                    file_name="review_analysis_results.csv",
                    mime="text/csv",
                )

# ── Tab 3: Semantic Search ────────────────────────────────────────────────────
from semantic_search import encode_reviews, search_reviews, get_similarity_label

with tab3:
    st.markdown(
        "Search reviews by **meaning**, not keywords. "
        "Try: *'battery issues'*, *'excellent sound quality'*, *'poor customer service'*"
    )

    st.subheader("Step 1: Load Reviews")
    search_source = st.radio(
        "Choose review source:",
        ["Use sample_reviews.csv", "Upload your own CSV"],
        horizontal=True,
    )

    reviews_for_search = []

    if search_source == "Use sample_reviews.csv":
        try:
            df_search = pd.read_csv("sample_reviews.csv")
            reviews_for_search = df_search["review"].dropna().tolist()
            st.success(f"Loaded {len(reviews_for_search)} sample reviews.")
        except FileNotFoundError:
            st.error("sample_reviews.csv not found in project folder.")
    else:
        uploaded_search = st.file_uploader(
            "Upload CSV with a 'review' column", type=["csv"], key="search_upload"
        )
        if uploaded_search:
            df_search = pd.read_csv(uploaded_search)
            if "review" not in df_search.columns:
                st.error("CSV must have a 'review' column.")
            else:
                reviews_for_search = df_search["review"].dropna().tolist()
                st.success(f"Loaded {len(reviews_for_search)} reviews.")

    if reviews_for_search:
        st.subheader("Step 2: Search")

        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g. battery drains too fast",
        )
        top_k = st.slider("Number of results", min_value=1, max_value=5, value=3)

        if st.button("Search", type="primary"):
            if not query.strip():
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Encoding reviews and searching..."):
                    embeddings = encode_reviews(reviews_for_search)
                    results = search_reviews(
                        query, reviews_for_search, embeddings, top_k=top_k
                    )

                st.subheader(f"Top {len(results)} results for: *'{query}'*")

                for r in results:
                    similarity_label = get_similarity_label(r["score"])
                    with st.expander(
                        f"#{r['rank']}  |  Similarity: {r['score']}%  {similarity_label}"
                    ):
                        st.write(r["review"])