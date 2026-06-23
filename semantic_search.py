"""
semantic_search.py
Semantic search over product reviews using SBERT (sentence-transformers).

Core idea:
  1. Encode all reviews into dense vector embeddings (done once)
  2. Encode the user's search query into a vector
  3. Find reviews whose vectors are closest to the query vector (cosine similarity)
  4. Return top-K most similar reviews
"""

from sentence_transformers import SentenceTransformer, util
import torch

# ── Load model (runs once at import) ─────────────────────────────────────────
print("Loading SBERT model...")
_sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
# 22MB model, downloads once and is cached locally
print("SBERT model loaded.")


# ── Core functions ────────────────────────────────────────────────────────────

def encode_reviews(reviews: list) -> torch.Tensor:
    """
    Convert a list of review strings into a matrix of embeddings.

    Each review  → 384-dimensional vector.
    10 reviews   → (10, 384) tensor matrix.

    Args:
        reviews: list of review strings

    Returns:
        torch.Tensor of shape (n_reviews, 384)
    """
    embeddings = _sbert_model.encode(
        reviews,
        convert_to_tensor=True,    # returns torch.Tensor (faster than numpy)
        show_progress_bar=False,
    )
    return embeddings


def search_reviews(
    query: str,
    reviews: list,
    embeddings: torch.Tensor,
    top_k: int = 3,
) -> list:
    """
    Find the top_k reviews most semantically similar to the query.

    Args:
        query:      user's search text (e.g. "battery drains too fast")
        reviews:    list of original review strings
        embeddings: pre-computed embeddings from encode_reviews()
        top_k:      number of results to return

    Returns:
        list of dicts: [{"rank": int, "review": str, "score": float}]
    """
    # Step 1: Encode the query into a 384-dim vector
    query_embedding = _sbert_model.encode(
        query,
        convert_to_tensor=True,
    )

    # Step 2: Compute cosine similarity between query and ALL review embeddings
    # Result shape: (1, n_reviews) → squeeze to (n_reviews,)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]

    # Step 3: Get top_k highest scoring indices
    top_k = min(top_k, len(reviews))
    top_results = torch.topk(cos_scores, k=top_k)

    # Step 4: Package results with human-readable scores
    results = []
    for rank, (score, idx) in enumerate(
        zip(top_results.values, top_results.indices), start=1
    ):
        results.append({
            "rank":   rank,
            "review": reviews[int(idx)],
            "score":  round(float(score) * 100, 1),  # 0.94 → 94.0%
        })

    return results


def get_similarity_label(score: float) -> str:
    """Human-readable label for a similarity score (0-100)."""
    if score >= 80:
        return "🟢 Very High"
    elif score >= 60:
        return "🟡 High"
    elif score >= 40:
        return "🟠 Moderate"
    else:
        return "🔴 Low"
