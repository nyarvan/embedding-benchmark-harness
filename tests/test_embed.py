"""Tests for Embedding: saved .npz layout and cosine search."""
import numpy as np
import pytest

from harness.embed import Embedding


def make_embedding() -> Embedding:
    embedding = Embedding(None)
    embedding.embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
    return embedding


def test_save_document_embeddings_creates_missing_directories(tmp_path):
    path = tmp_path / 'scifact-500' / 'model' / 'documents.npz'

    make_embedding().save_document_embeddings(path, np.array(['a', 'b', 'c']))

    assert path.exists()


def test_save_document_embeddings_stores_ids_and_embeddings(tmp_path):
    path = tmp_path / 'documents.npz'
    ids = np.array(['a', 'b', 'c'])
    embedding = make_embedding()

    embedding.save_document_embeddings(path, ids)

    saved = np.load(path)
    assert set(saved.files) == {'document_ids', 'embeddings'}
    assert saved['document_ids'].tolist() == ['a', 'b', 'c']
    assert np.array_equal(saved['embeddings'], embedding.embeddings)


def test_save_chunk_embeddings_stores_chunk_and_document_ids(tmp_path):
    path = tmp_path / 'nested' / 'chunks.npz'

    make_embedding().save_chunk_embeddings(
        path,
        chunk_ids=np.array(['a_0', 'a_1', 'b_0']),
        document_ids=np.array(['a', 'a', 'b']),
    )

    saved = np.load(path)
    assert set(saved.files) == {'chunk_ids', 'document_ids', 'embeddings'}
    assert saved['chunk_ids'].tolist() == ['a_0', 'a_1', 'b_0']
    assert saved['document_ids'].tolist() == ['a', 'a', 'b']


def test_save_chunk_embeddings_requires_ids_as_keywords(tmp_path):
    ids = np.array(['a_0', 'a_1', 'b_0'])

    with pytest.raises(TypeError):
        make_embedding().save_chunk_embeddings(tmp_path / 'c.npz', ids, ids)


def test_save_before_encode_raises_runtime_error(tmp_path):
    ids = np.array(['a'])

    with pytest.raises(RuntimeError):
        Embedding(None).save_document_embeddings(tmp_path / 'd.npz', ids)
    with pytest.raises(RuntimeError):
        Embedding(None).save_chunk_embeddings(
            tmp_path / 'c.npz', chunk_ids=ids, document_ids=ids
        )


QUERIES = {
    'embeddings': np.array([[1.0, 0.0]], dtype=np.float32),
    'document_ids': np.array(['q1']),
}
VECTORS = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)


def test_search_returns_exactly_top_k_documents_best_first():
    documents = {
        'embeddings': VECTORS, 'document_ids': np.array(['d0', 'd1', 'd2']),
    }

    result = Embedding(None).search(QUERIES, documents, top_k=2)

    # Guards against passing top_k positionally: the third positional
    # parameter of semantic_search is query_chunk_size, which once made every
    # search return only 10 results
    assert list(result['q1']) == ['d0', 'd2']


def test_search_chunks_returns_exactly_top_k_chunks_best_first():
    chunks = {
        'embeddings': VECTORS,
        'chunk_ids': np.array(['d0_0', 'd1_0', 'd1_1']),
        'document_ids': np.array(['d0', 'd1', 'd1']),
    }

    result = Embedding(None).search_chunks(QUERIES, chunks, top_k=2)

    assert list(result['q1']) == ['d0_0', 'd1_1']


def test_search_scores_are_cosine_similarities():
    documents = {
        'embeddings': VECTORS, 'document_ids': np.array(['d0', 'd1', 'd2']),
    }

    result = Embedding(None).search(QUERIES, documents, top_k=3)

    assert result['q1']['d0'] == pytest.approx(1.0)
    assert result['q1']['d2'] == pytest.approx(np.sqrt(0.5))
    assert result['q1']['d1'] == pytest.approx(0.0, abs=1e-6)
