# InvestRAG Studio

**A document-intelligence and RAG evaluation platform that reports how each answer was
produced — including when no model wrote it.**

Ingest mixed-format research material (PDF, DOCX, PPTX, XLSX, HTML, EML, MSG, images), keep
every version immutable, retrieve over it with a hybrid dense + BM25 + RRF + cross-encoder
pipeline, and score the whole thing against a golden set with metrics validated against an
independent reference implementation.

[![CI](https://github.com/Lekh007/investrag-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/Lekh007/investrag-studio/actions/workflows/ci.yml)

![Data Room: the original PDF page beside the extracted structure, with each element's real
bounding box drawn over it](docs/evidence/phase-8/04-dataroom-bbox-overlay.png)

*Every citation opens the exact page with its bounding box highlighted. Note the source
registry above it: `queryable`, `failed` and `review required` are real states the system
distinguishes — a low-confidence extraction is inspectable but never retrievable evidence
until an operator accepts it.*

## Measured, not asserted

On a reproducible 15-question golden set spanning all seven question classes — direct-fact,
numerical, exact-identifier, multi-hop, metadata-filter, conflicting and unanswerable:

| | |
|---|---|
| Success@5 | **1.0** |
| nDCG@5 | **0.848** |
| Abstention accuracy | **1.0** |
| Agreement with `ir-measures` | **within 1e-6** |

That last row is the point. The IR metrics are hand-implemented, so they are cross-checked
against the independent reference library — which proves the formulas are *correct*, not merely
self-consistent.

Building that set found two real bugs in the retrieval and evaluation path, both fixed with
regression tests:

- **A metric that punished correct abstention.** An unanswerable question has no relevant chunks
  by construction, so Success@k / Recall@k / nDCG@k are mathematically 0 for it no matter how
  good retrieval was. Averaging those in scored a perfect 13-of-13 answerable run as **86.7%**.
  These metrics are now computed only over answerable questions, with abstention accuracy kept as
  the separate, correct gate for the unanswerable class.
- **An evidence gate that passed on a coincidence.** One stopword-filtered term match — "current"
  in a loan's repayment status matching "the **current** price of gold" — was enough to bless
  every candidate for a completely unrelated question. The gate now requires two overlapping
  terms for multi-term queries.

## Every answer says how it was produced

The degraded paths are otherwise indistinguishable from a working one: the extractive fallback
emits `[S1]` labels and therefore satisfies citation validation exactly like a real completion.
So `trace.generation_mode` is always one of:

| Mode | Meaning |
|------|---------|
| `model` | the local model wrote the answer and its citations resolve |
| `extractive-fallback` | the model was unreachable; the text is verbatim retrieved evidence, labelled as such in the answer body, a validator message and the UI badge |
| `abstained` | retrieval found no supporting evidence, so **the model was never called** |

`abstained` is load-bearing. Asked something the corpus does not cover, a local model will answer
from parametric memory and invent both a figure and an `[S1]` label to go with it — measured, not
hypothesised (llama3.1:8b, 2026-08-29: a fabricated gold price). Abstention is therefore decided
by retrieval, never by the model.

## Seven vector stores, six verified live

FAISS, Chroma, Qdrant and Milvus run **embedded** — no service, no Docker. Weaviate needs Docker,
pgvector needs the included Postgres profile, Pinecone needs an API key. Six of the seven were
each round-tripped with a real chunk, provenance and a `source_id` filter through a real running
instance on 2026-08-30; only Pinecone is implemented against documented API without a live index.

Milvus needed an actual fix rather than a flag: `pymilvus`'s own package metadata excludes `win32`
from its `milvus_lite` extra — a stale marker, since the wheel has supported Windows since 3.2.1.
This project depends on `milvus-lite` directly instead of through that broken extra.

## Run it

```powershell
cd backend
uv sync
$env:INVESTRAG_EMBEDDING_MODE = "hash"   # offline smoke test; omit for real BGE-M3
uv run uvicorn investrag.main:app --host 127.0.0.1 --port 8000

cd ../frontend
npm install && npm run dev               # http://127.0.0.1:5174
```

The full gates — `ruff`, `mypy --strict`, `pytest` (201 passing), `vitest` (47), production
build — run in [CI](.github/workflows/ci.yml) on every push and need no model download, vector
service or database.

## Honest scope

The corpus behind the numbers above is **fixture-backed, not real filings** — the documents in the
screenshot are generated. The metrics are real and reproducible; the documents they are measured
over are synthetic, which is a deliberate tradeoff for a reproducible gate and a real limitation
when reading the results.

Conflicting-evidence detection is topic-level (shared vocabulary plus differing numbers with the
same unit), not claim-level entailment. MinerU escalation is detected and recorded but not acted
on — there is a tested port and preflight, but no worker is provisioned, because a real
`pip install --dry-run` confirmed it cannot co-install with FastAPI's dependency tree.

This is a tested local vertical slice, not a production system, and not investment advice.

## Going deeper

**[DEEP_DIVE.md](DEEP_DIVE.md)** has the complete version: every capability in the current slice,
the parser bake-off that set the escalation thresholds, the security trust boundary and the live
prompt-injection hole it closed, the golden-set benchmark flow, the acceptance exercise, and the
scope that is explicitly still open.
