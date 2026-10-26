import pytest

from domain.chunk import Chunking
from harness.loaders import load_beir_dataset


text = '''
    Alterations of the architecture of cerebral
    white matter in the developing human brain
    can affect cortical development and result in
    functional disabilities. A line scan diffusion-weighted magnetic resonance
    imaging (MRI) sequence with diffusion tensor analysis was applied to
    measure the apparent diffusion coefficient, to calculate relative
    anisotropy, and to delineate three-dimensional fiber architecture in
    cerebral white matter in preterm (n = 17) and full-term infants (n = 7).
    To assess effects of prematurity on cerebral white matter development,
    early gestation preterm infants (n = 10) were studied a second time at
    term.
    In the central white matter the mean apparent diffusion coefficient at 28
    wk
    was high, 1.8 microm2/ms, and decreased toward term to 1.2 microm2/ms.
    In the posterior limb of the internal capsule, the mean apparent diffusion
    coefficients at both times were similar (1.2 versus 1.1 microm2/ms).
    Relative anisotropy was higher the closer birth was to term with greater
    absolute values in the internal capsule than in the central white matter.
    Preterm infants at term showed higher mean diffusion coefficients in the
    central white matter (1.4 +/- 0.24 versus 1.15 +/- 0.09 microm2/ms,
    p = 0.016) and lower relative anisotropy in both areas compared with
    full-term infants (white matter, 10.9 +/- 0.6 versus 22.9 +/- 3.0%,
    p = 0.001; internal capsule, 24.0 +/- 4.44 versus 33.1 +/- 0.6% p = 0.006).
    Nonmyelinated fibers in the corpus callosum were visible by diffusion
    tensor
    MRI as early as 28 wk; full-term and preterm infants at term showed marked
    differences in white matter fiber organization. The data indicate that
    quantitative assessment of water diffusion by diffusion tensor MRI provides
    insight into microstructural development in cerebral white matter in
    living infants.
    '''


def make_chunking(size: int = 500, overlap: int = 0):
    return Chunking(size, overlap)


@pytest.fixture(scope='module')
def scifact_docs():
    docs, _, _ = load_beir_dataset(split='train')
    return docs


def test_fixed_chunker_joined_chunks_equal_original_text():
    chunking = make_chunking()
    assert ''.join(chunking.fixed_chunker(text)) == text


def test_fixed_chunker_all_but_last_chunk_have_exact_size():
    chunking = make_chunking()
    size = 500
    chunks = chunking.fixed_chunker(text)

    result = True

    for chunk in chunks[:-1]:
        if len(chunk) != size:
            result = False
            break

    assert result is True


def test_fixed_chunker_short_one_chunk_and_empty_empty():
    chunking = make_chunking()
    small_size = int(chunking.size / 2)
    short = text[:small_size]
    empty = ''

    short_chunks = chunking.fixed_chunker(short)
    empty_chunks = chunking.fixed_chunker(empty)

    assert len(short_chunks) == 1
    assert empty_chunks == []


def test_fixed_chunker_with_overlap_returns_expected_chunks():
    chunking = make_chunking(size=5, overlap=2)
    t = 'abcdefghij'
    chunks = chunking.fixed_chunker(t)

    assert chunks == ["abcde", "defgh", "ghij"]


def test_fixed_chunker_invalid_size_or_overlap_raises_value_error():
    params = [(0, 0), (500, -1), (500, 1000)]

    for s, o in params:
        chunking = make_chunking(s, o)
        with pytest.raises(ValueError):
            chunking.fixed_chunker(text)


def test_chunk_documents_keeps_document_id_and_unique_chunk_ids():
    docs = [('d1', {'text': 'a' * 1200}), ('d2', {'text': 'b' * 300})]

    chunking = make_chunking()

    chunks = chunking.chunk_documents(docs, chunking.fixed_chunker)

    doc_id, doc = docs[0]

    doc_chunks = [chunk for chunk in chunks if chunk.document_id == doc_id]

    f_text = ''.join([chunk.text for chunk in doc_chunks])

    ids = [chunk.chunk_id for chunk in chunks]

    assert chunks[0].chunk_id == f'{doc_id}_0'
    assert len(ids) == len(set(ids))
    assert f_text == doc['text']


def test_recursive_chunk_less_size():
    t = 'aaa\n\nbbb\n\nccc'
    chunking = make_chunking(size=8)

    chunks = chunking.recursive_chunker(t)

    result = True

    for chunk in chunks:
        if len(chunk) > chunking.size:
            result = False
            break

    assert result is True


def test_correct_recursive_chunk():
    chunking = make_chunking(size=7)
    t = 'a. b.\n\nc. d.'
    chunks = chunking.recursive_chunker(t)

    assert chunks == ['a. b.', 'c. d.']


def test_not_empty_chunks_recursive_chunk():
    chunking = make_chunking()
    chunks = chunking.recursive_chunker(text)

    result = True
    for chunk in chunks:
        if chunk == '':
            result = False
            break

    assert result is True


def test_short_empty_text_recursive_chunk():
    chunking = make_chunking()
    short = 'a' * 250
    empty = ''

    chunks1 = chunking.recursive_chunker(short)
    chunks2 = chunking.recursive_chunker(empty)

    assert len(chunks1) == 1
    assert chunks2 == []


@pytest.mark.integration
@pytest.mark.parametrize('method', ['fixed_chunker', 'recursive_chunker'])
def test_chunkers_on_scifact_keep_size_limit_and_no_empty_chunks(
    scifact_docs, method
):
    chunking = make_chunking()
    chunks = chunking.chunk_documents(scifact_docs, getattr(chunking, method))

    assert all(0 < len(c.text) <= 500 for c in chunks)
    assert {c.document_id for c in chunks} == {
        d_id for d_id, _ in scifact_docs
    }
