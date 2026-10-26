# embedding-benchmark-harness

A small, reproducible harness for comparing embedding models and document
chunking strategies on retrieval benchmarks with ready-made relevance
judgments ([BEIR](https://github.com/beir-cellar/beir)).

It measures **P@5, Recall@10 and NDCG@10**, and goes one step further than a
single averaged number: per-query scores, paired bootstrap confidence
intervals, and a tool to inspect the queries where one strategy beats another.

Current scope: **SciFact**, a 500-document subset, local open-source models
(`BAAI/bge-m3`, `intfloat/e5-large-v2`) via `sentence-transformers` on CPU.

---

## Results

### 1. Model comparison (whole documents)

Each document is embedded as `Title: {title}\nText: {text}`. 809 queries from
the SciFact `train` split, 500 documents.

| Model | P@5 | Recall@10 | NDCG@10 |
|---|---|---|---|
| `bge-m3` | 0.0215 | 0.1023 | 0.0933 |
| `e5-large-v2` | 0.0218 | 0.1027 | 0.0967 |

The two models are practically indistinguishable on this subset (see
[Why the numbers are low](#why-the-numbers-are-low)).

### 2. Chunking experiment (`bge-m3`)

Documents are split into chunks of at most **500 characters**, overlap 0.
Every chunk is embedded with its document title in the same `Title/Text`
format as the baseline, so chunking is the only thing that changes.

| Strategy | Chunks | P@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| No chunking (baseline) | 500 | 0.0215 | 0.1023 | 0.0933 |
| Fixed-size | 1639 | 0.0220 | 0.1027 | 0.0946 |
| Recursive | 1842 | 0.0220 | **0.1052** | **0.0955** |

Rescaled to the 99 queries that can score above zero at all (see below):

| Strategy | P@5 | Recall@10 | NDCG@10 |
|---|---|---|---|
| No chunking | 0.176 | 0.836 | 0.763 |
| Fixed-size | 0.180 | 0.839 | 0.773 |
| Recursive | 0.180 | **0.859** | **0.781** |

### 3. Is the difference real?

Paired bootstrap over the 99 active queries (10,000 resamples, seed 0).
Differences are `strategy − baseline`, positive means chunking is better.

| Comparison | ΔNDCG@10 | 95% CI | ΔRecall@10 | 95% CI |
|---|---|---|---|---|
| Fixed − baseline | +0.010 | [−0.020, +0.038] | +0.003 | [−0.037, +0.044] |
| Recursive − baseline | +0.018 | [−0.005, +0.041] | +0.024 | [0.000, +0.054] |
| Recursive − fixed | +0.008 | [−0.010, +0.028] | +0.020 | [0.000, +0.051] |

How many of the 99 queries changed:

| Strategy vs baseline | NDCG@10 better / worse / same | Recall@10 better / worse |
|---|---|---|
| Fixed-size | 13 / 7 / 79 | 3 / 2 |
| Recursive | 13 / 4 / 82 | 3 / 0 |

Every confidence interval includes zero. The Recall intervals of recursive end
exactly at 0.0 because only 2–3 queries changed, all in its favour. An exact
two-sided sign test on the changed queries gives p ≈ 0.05 for recursive vs
baseline on NDCG@10 (13 vs 4), p ≈ 0.26 for fixed (13 vs 7) and p = 0.25 for
recursive Recall@10 (3 vs 0). These tests are post hoc and uncorrected for
multiple comparisons.

### 4. What the queries show

Queries where chunking helped (rank of the relevant document):

| Query | Baseline | Fixed | Recursive | Best matching chunk |
|---|---|---|---|---|
| *Deleting Raptor increases G-CSF levels.* | 14 | 3 | 4 | middle of the abstract, mentions G-CSF |
| *Ribosomal protein (RP) expression is controlled in part by stress-activated regulators.* | 11 | 4 | 5 | first chunk, "stress-activated regulatory network" |
| *Sudden death can occur in patients with complex ventricular arrhythmias.* | 17 | 8 | 7 | first chunk, "associated with sudden death" |

Queries where recursive chunking hurt the most: the relevant document dropped
from rank 1 to rank 2–3 (queries 95, 676, 1294). It never left the top 10, so
Recall@10 did not suffer.

### Conclusion

On SciFact with `bge-m3`, chunking into 500-character pieces gives **no
statistically significant gain**. Recursive chunking is nominally the best
variant on every metric and hurts fewer queries than fixed-size chunking
(4 vs 7), but the improvement rests on a handful of queries.

The pattern behind it is consistent: SciFact queries are single claims,
while abstracts cover several topics. A chunk around the matching sentence is
closer to the claim than the whole abstract, which lifts documents from ranks
11–17 into the top 10. The price is small reshuffling at the top (1 → 2–3)
when chunks of other documents happen to match better. With abstracts of
~300 tokens, well within the model context window, there is little for
chunking to rescue.

Two caveats:

- Fixed and recursive chunking lifted the same documents almost equally, so
  the effect comes from **shorter fragments**, not from cleaner boundaries.
- The title is prepended to every chunk and weighs more in a short chunk. Part
  of the gain may come from the title, not from the matching sentence. This is
  not tested yet (see [Roadmap](#roadmap)).

### Why the numbers are low

The corpus is a 500-document subset of SciFact's 5,183, but the qrels and
queries are not filtered. Only **99 of 809 queries** have a relevant document
inside the subset; the other 710 score zero for every model and strategy and
divide every average by 809. The rescaled tables above show the real level:
`bge-m3` finds the relevant document in the top 10 for ~84% of the queries
that have one. The comparison between strategies stays fair, but the sample
is small, which is why the differences are hard to prove.

---

## How it works

```text
documents ─┬─────────────────────────────► embed ─┐
           │                                      │
           └─► chunk (fixed | recursive) ─► embed ─┤
                                                  ▼
queries ──────────────────────────────────► cosine search
                                                  │
              chunk ids ─► max-pool by document ◄─┘ (chunk strategies only)
                                                  │
                                                  ▼
                              document ranking ─► BEIR evaluation (qrels)
```

**Chunk → document.** Qrels refer to documents, retrieval over chunks returns
chunks. Every chunk keeps its `document_id` (chunk ids look like
`<document_id>_<index>`). For each query the harness retrieves the top 50
chunks, scores each document by its best chunk (max-pooling), sorts and keeps
the top 10 documents. The evaluation code is the same for all variants.

**Chunkers** (`domain/chunk.py`):

- `fixed_chunker` cuts the text into windows of `size` characters with
  optional overlap; joining the chunks restores the text exactly.
- `recursive_chunker` splits by paragraphs, then lines, sentences and words,
  recursing only into pieces that are still too long, and greedily merges
  small neighbours back up to `size`.

The chunk size of 500 characters comes from the length distribution of the
subset (`text_lengths.py`): median 1,320, mean 1,378, p90 2,001, max 4,963
characters. At 500, 98% of documents are split (about 3.3 chunks each); at
2,000 only 10% would be, which would make the experiment meaningless.

---

## Project structure

```text
embedding-benchmark-harness/
├── harness/
│   ├── loaders.py       # BEIR download/loading, metrics → JSON
│   ├── embed.py         # encoding, .npz persistence, cosine search
│   ├── aggregate.py     # chunk ranking → document ranking (max-pooling)
│   └── evaluate.py      # P@5, Recall@10, NDCG@10 via BEIR
├── domain/
│   └── chunk.py         # Chunk, fixed and recursive chunkers
├── main.py              # build/cache embeddings, retrieve, evaluate, save
├── significance.py      # per-query scores, paired bootstrap
├── inspect_losers.py    # win/loss counts, query-level inspection
├── text_lengths.py      # length statistics for choosing the chunk size
├── tests/               # pytest: chunkers, aggregation
├── data/                # cached embeddings (.npz), generated
├── datasets/            # BEIR downloads, generated
└── results/             # metrics as JSON
```

---

## Getting started

Developed and tested on Python 3.14; `requirements.txt` pins the exact
versions of that environment. CPU is enough. The first run downloads SciFact
(~3 MB) and the models (`bge-m3` is ~2.3 GB).

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the benchmark

```bash
python main.py
```

Builds the document, query and chunk embeddings for every model in
`MODEL_IDS`, retrieves, evaluates and writes
`results/scifact_500docs_chunking_topk200.json`.

Embeddings are cached in `data/` and **an existing file is never
recomputed**. Delete the file after changing the model, the chunk size or the
input format; otherwise stale embeddings are silently reused.

### Analyse the results

```bash
python significance.py     # verifies per-query scores against the saved table,
                           # prints bootstrap intervals and changed queries
python inspect_losers.py   # win/loss counts and the worst queries in detail
python text_lengths.py     # document length statistics
```

### Tests

```bash
pytest -m "not integration"   # fast unit tests
pytest -m integration         # chunkers on the real SciFact subset
pytest                        # everything
```

Run commands from the repository root: data paths are relative to it.

---

## Lessons learned

Three bugs from building this harness, each of which produced plausible but
wrong numbers:

1. **`util.semantic_search(q, docs, top_k)` ignores `top_k`.** The third
   positional parameter is `query_chunk_size`; `top_k` must be passed by
   keyword. Whole-document search was unaffected only because the wanted
   value happened to equal the default of 10. Chunk search silently returned
   10 chunks instead of 50.
2. **Too few chunks before aggregation cap Recall@10.** Ten chunks collapsed
   into ~6.7 distinct documents on average, so 98% of queries had fewer than
   10 documents to evaluate. The chunk-level `top_k` must be well above the
   document-level cut-off; here, results are identical for 50, 200 and all
   chunks.
3. **Swapped id arrays fail silently.** Chunk ids and document ids are both
   string arrays; passing them in the wrong order produces a valid file and
   meaningless metrics. The save method now takes them as keyword-only
   arguments.

A general check that caught problems early: the whole-document variant must
reproduce the previous results to the last digit before any comparison is
trusted.

---

## Known limitations

- 500-document subset with unfiltered qrels: only 99 informative queries.
- `e5-large-v2` is used without its recommended `query: ` / `passage: `
  prefixes, so its numbers are likely below its potential.
- One chunk size (500 characters) and no overlap.

## Roadmap

- [ ] OpenAI `text-embedding-3-small` as a third model
- [ ] Chunks without the title, to separate the title effect from chunking
- [ ] Full SciFact corpus (5,183 documents) for a larger effective sample
- [ ] `e5` with query/passage prefixes; chunking experiment on `e5`
- [ ] CLI: `python cli.py run --dataset scifact --models bge-m3,e5 --chunking fixed,recursive`
- [ ] Unit tests for `loaders.py` and `evaluate.py`
- [ ] Hybrid search (BM25 + dense) and reranking, as separate experiments

## License

[MIT](LICENSE) © 2026 Ivan Rybak
