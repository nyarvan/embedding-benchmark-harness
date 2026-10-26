"""Retrieval metrics computed with BEIR."""
from beir.retrieval.evaluation import EvaluateRetrieval


def evaluate(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute P@5, Recall@10 and NDCG@10 averaged over the queries.

    Args:
        qrels: {query_id: {doc_id: relevance}}.
        results: {query_id: {doc_id: score}}. BEIR ranks documents by score
            and applies the cut-off k itself.
    """
    retriever = EvaluateRetrieval()
    ndcg, _, recall, precision = retriever.evaluate(
        qrels,
        results,
        retriever.k_values,
    )
    return {
        'P@5': precision["P@5"],
        'Recall@10': recall["Recall@10"],
        'NDCG@10': ndcg["NDCG@10"]
    }
