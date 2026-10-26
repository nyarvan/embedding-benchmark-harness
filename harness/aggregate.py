"""Aggregation of chunk-level rankings into document-level rankings."""


def aggregate_to_documents(
    chunk_results: dict[str, dict[str, float]], top_k: int = 10
) -> dict[str, dict[str, float]]:
    """Turn chunk rankings into document rankings for every query.

    A document is scored by its best chunk (max-pooling), so qrels, which are
    defined on documents, can be applied to chunk-level retrieval.

    Args:
        chunk_results: {query_id: {chunk_id: score}}, chunk ids look like
            '<document_id>_<chunk_index>'.
        top_k: number of documents kept per query. Search chunk-level with a
            larger top_k than this: several chunks of one document collapse
            into a single document.

    Returns:
        {query_id: {document_id: score}}, ordered by descending score.
    """
    result = {}
    for q_id, chunk_scores in chunk_results.items():
        result[q_id] = _max_pool_by_document(chunk_scores, top_k)
    return result


def _max_pool_by_document(
    chunk_scores: dict[str, float], top_k: int
) -> dict[str, float]:
    doc_scores = {}
    for chunk_id, score in chunk_scores.items():
        # Document ids may contain '_', the chunk index is always the last part
        doc = chunk_id.rsplit('_', 1)[0]
        if doc not in doc_scores or score > doc_scores[doc]:
            doc_scores[doc] = score
    ranked = sorted(
        doc_scores.items(), key=lambda item: item[1], reverse=True
    )[:top_k]
    return dict(ranked)
