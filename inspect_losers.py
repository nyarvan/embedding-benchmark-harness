"""Inspect queries where chunking wins or loses against the baseline."""
import sys

import numpy as np

from domain.chunk import Chunking
from harness.aggregate import aggregate_to_documents
from harness.embed import Embedding
from harness.loaders import load_beir_dataset
from significance import (
    get_active_queries, per_query_scores, retrieve_documents,
)

STRATEGY_FILES = {
    'baseline': 'embeddings',
    'fixed': 'chunks_fixed',
    'recursive': 'chunks_recursive',
}
EPS = 1e-9  # scores closer than this count as equal


def count_changes(scores: dict[str, dict[str, np.ndarray]]) -> None:
    """Print how many queries each chunking strategy improves or hurts."""
    for strategy in ('fixed', 'recursive'):
        for metric in ('NDCG@10', 'Recall@10'):
            diff = scores[strategy][metric] - scores['baseline'][metric]
            print(
                f'{strategy} {metric}: better={int((diff > EPS).sum())} '
                f'worse={int((diff < -EPS).sum())} '
                f'same={int((abs(diff) <= EPS).sum())}'
            )


def worst_queries(
    scores: dict[str, dict[str, np.ndarray]], active: list[str],
    count: int = 4,
) -> list[tuple[str, float]]:
    """Queries where recursive loses most NDCG@10 against the baseline."""
    diff = scores['recursive']['NDCG@10'] - scores['baseline']['NDCG@10']
    order = np.argsort(diff)[:count]
    return [(active[i], float(diff[i])) for i in order if diff[i] < -EPS]


def rank_of(doc_id: str, ranking: dict[str, float]) -> int | None:
    """1-based position of doc_id in a ranking, or None if absent.

    `ranking` must be ordered best first (search results and aggregation
    already are).
    """
    for position, ranked_id in enumerate(ranking, start=1):
        if ranked_id == doc_id:
            return position
    return None


def best_chunk(
    doc_id: str, chunk_scores: dict[str, float], chunk_texts: dict[str, str],
) -> tuple[str, float, str]:
    """(chunk_id, score, text) of the document's highest-scoring chunk.

    `chunk_scores` must contain at least one chunk of the document.
    """
    own = {
        chunk_id: score for chunk_id, score in chunk_scores.items()
        if chunk_id.rsplit('_', 1)[0] == doc_id
    }
    chunk_id = max(own, key=own.get)
    return chunk_id, own[chunk_id], chunk_texts[chunk_id]


def _shorten(text: str, limit: int = 300) -> str:
    """Collapse whitespace and cut the text to `limit` characters."""
    flat = ' '.join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + '...'


def inspect_queries(q_ids: list[str], name: str, size: int = 500) -> None:
    """Print, per query, ranks of relevant docs and their best chunks.

    Chunk texts are rebuilt with the same chunker settings as the saved chunk
    embeddings (`size`, overlap 0); a mismatch in chunk count raises.
    """
    documents, queries, qrels = load_beir_dataset(split='train')
    corpus = dict(documents)
    q_emb = np.load(f'data/embeddings_queries_{name}.npz')
    embedding = Embedding(None)
    chunker = Chunking(size)
    splitters = {
        'fixed': chunker.fixed_chunker,
        'recursive': chunker.recursive_chunker,
    }

    # Search the whole corpus so that every relevant document gets a rank
    rankings = {
        'baseline': embedding.search(
            q_emb, np.load(f'data/embeddings_{name}.npz'), top_k=len(corpus)
        )
    }
    chunk_results, chunk_texts = {}, {}
    for strategy, splitter in splitters.items():
        chunks = np.load(f'data/chunks_{strategy}_{name}.npz')
        chunk_results[strategy] = embedding.search_chunks(
            q_emb, chunks, top_k=len(chunks['embeddings'])
        )
        rankings[strategy] = aggregate_to_documents(
            chunk_results[strategy], len(corpus)
        )
        chunk_texts[strategy] = {
            c.chunk_id: c.text
            for c in chunker.chunk_documents(documents, splitter)
        }
        if len(chunk_texts[strategy]) != len(chunks['chunk_ids']):
            raise ValueError(f'{strategy}: chunk texts do not match the file')

    for q_id in q_ids:
        print(f'\n=== Query {q_id}: {queries[q_id]}')
        relevant = [
            d for d, score in qrels[q_id].items() if score > 0 and d in corpus
        ]
        for doc_id in relevant:
            ranks = {s: rank_of(doc_id, rankings[s][q_id]) for s in rankings}
            print(f'\n  doc {doc_id}: {corpus[doc_id]["title"]}')
            print(f'    ranks: {ranks}')
            print(f'    start: {_shorten(corpus[doc_id]["text"], 200)}')
            for strategy in splitters:
                chunk_id, score, text = best_chunk(
                    doc_id, chunk_results[strategy][q_id],
                    chunk_texts[strategy],
                )
                print(f'    best {strategy} chunk {chunk_id} ({score:.3f}): '
                      f'{_shorten(text)}')


def main(name: str = 'bge_m3', top_k: int = 50) -> None:
    """Print win/loss counts and inspect the worst queries of recursive."""
    documents, _, qrels = load_beir_dataset(split='train')
    active = get_active_queries(qrels, documents)
    queries = np.load(f'data/embeddings_queries_{name}.npz')

    scores = {
        strategy: per_query_scores(
            qrels,
            retrieve_documents(
                strategy, queries, f'data/{file}_{name}.npz', top_k
            ),
            active,
        )
        for strategy, file in STRATEGY_FILES.items()
    }

    count_changes(scores)

    losers = worst_queries(scores, active)
    print('\nworst recursive vs baseline (NDCG):',
          [(q, round(diff, 3)) for q, diff in losers])

    gained = [
        q for q, fixed, recursive in zip(
            active, scores['fixed']['Recall@10'],
            scores['recursive']['Recall@10'],
        )
        if recursive - fixed > EPS
    ]
    print('recall gained by recursive over fixed:', gained)

    inspect_queries([q for q, _ in losers[:3]], name)


if __name__ == '__main__':
    # Usage: python inspect_losers.py [model_name], default bge_m3
    main(sys.argv[1] if len(sys.argv) > 1 else 'bge_m3')
