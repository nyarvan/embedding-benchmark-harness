"""Dataset loading and metrics persistence."""
import json
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader


def load_beir_dataset(
    name: str = 'scifact', split: str = 'test', size: int = 500
) -> tuple[
    list[tuple[str, dict[str, str]]], dict[str, str], dict[str, dict[str, int]]
]:
    """Download (once) and load a BEIR dataset.

    Returns:
        documents: the first `size` corpus items as (doc_id, {'title', 'text'}).
        queries: {query_id: query_text} for the split.
        qrels: {query_id: {doc_id: relevance}} for the split. They cover the
            whole corpus, not only the `size` documents kept here.
    """
    url = (
        'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/'
        f'{name}.zip'
    )
    data_path = util.download_and_unzip(url, 'datasets')
    corpus, queries, qrels = GenericDataLoader(data_path).load(split)
    documents = list(corpus.items())[:size]
    return documents, queries, qrels


def save_metrics(
    retrieval_results: list[tuple[str, dict[str, float]]],
    output_path: str | Path,
) -> None:
    """Write (run_name, metrics) pairs as a JSON list of {run_name: metrics}."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for run_name, results in retrieval_results:
        records.append({
            run_name: {
                'P@5': results['P@5'],
                'Recall@10': results['Recall@10'],
                'NDCG@10': results['NDCG@10']
            }
        })

    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8'
    )
