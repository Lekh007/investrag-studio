# InvestRAG Studio — deep dive

The [README](README.md) is the short version. This is the complete one: every
capability in the current slice, the parser bake-off that set the escalation
thresholds, the security trust boundary and the live prompt-injection hole it
closed, the golden-set benchmark flow, and the scope that is explicitly still
open.

## Current vertical slice

- FastAPI backend with logical-source grouping and append-only immutable versions, behind a `CatalogPort` with both a SQLite adapter (network-free default) and a real PostgreSQL adapter (`postgres_dsn`), both proven interchangeable against the same contract test;
- content-signature-aware Markdown, TXT, HTML, JSON, CSV, XLSX, PDF, DOCX, PPTX, EML, Outlook MSG and image routing;
- Docling as the real primary layout parser for PDF/DOCX/PPTX/HTML (walks the actual document item tree, not a markdown round-trip, so bounding boxes and per-row table cells survive), with a PyMuPDF4LLM second opinion whose page-level disagreement score and a pure escalation-policy function (`escalation.py`) decide when a page needs MinerU review — MinerU itself runs out-of-process behind a resource-profile preflight (`mineru_port.py`, `resources.py`) since it is not co-installable with FastAPI's dependency tree (verified, see "Parser bake-off" below);
- real Tesseract OCR with per-word confidence (`ocr.py`) when the optional `parsers` extra and the Tesseract binary are installed, replacing hardcoded confidence placeholders;
- canonical elements with page, slide, sheet, cell-range, JSONPath and bounding-box provenance;
- four chunk profiles — structure-aware, recursive-baseline, parent-child (real full-section parent chunks a child's `parent_id` resolves to) and table-aware (table summary + header-carrying row-group chunks) — listed with versions at `/chunk-profiles`;
- BGE-M3 embeddings when the local model is available, with a deterministic hash fallback for offline tests;
- persistent FAISS cosine index;
- BM25 plus reciprocal-rank fusion through the LangChain Runnable used by the serving path, followed by real cross-encoder reranking (`mixedbread-ai/mxbai-rerank-base-v2`) over a widened top-20 pool before truncating to the requested top_k — see "Retrieval pipeline" below for why this model was chosen over the design's original `bge-reranker-v2-m3`;
- an allowlisted metadata-filter derivation pass (`filters.py`) that extracts in-query hints like `page:3` or `sheet:Summary`, validates them, and rewrites the query text with the hint removed before it reaches embedding/BM25 — quoted phrases are never rewritten;
- LLM-generated multi-query expansion (`granite4.2:3b`) with a deterministic three-template fallback, both paths visible in the trace;
- a `parent` retrieval profile that expands a search hit to its real, full-section parent chunk (from the `parent-child` chunk profile) rather than only deduplicating;
- an evidence packet bounded by `granite4.2:3b`'s real tokenizer, not a character count (`token_budget.py`), with an offline chars-per-token fallback;
- local Ollama answer generation with declared provenance (see below) and a citation-repair retry;
- a persistent embedding cache keyed by chunk-text checksum + embedding-model version — re-ingesting an unchanged corpus performs zero embedding recomputation;
- persisted query traces (`GET /queries/traces`, `GET /queries/traces/{trace_id}`) so retrieval history survives past the single most-recent query;
- citation-bearing answers, citation-label validation and measured per-stage query traces;
- **job-based ingestion with server-sent stage progress** (design §16, §8 step 14) — `POST /ingestions` returns `202` with a job id and a `Location` header instead of holding the request open, and `GET /ingestions/{id}/events` streams one frame per §8 stage with the real detail each one discovered; a late subscriber first receives a `snapshot` of everything it missed, and a failed job keeps its completed stages and can be resumed from the staged artifact;
- **reprocessing with an alternative parser or chunk profile** (`POST /source-versions/{id}/reprocess`, design §15.2) — re-runs the original bytes through the same pipeline and produces a *new* immutable version (both overrides are part of the version hash), never mutating the one it came from; chunk profile and parser mode are also selectable per upload;
- **an operator review gate** (`POST /source-versions/{id}/review`, design §15.6) — a low-confidence extraction is inspectable but is never retrievable evidence until an operator accepts it, and the override is recorded on the source;
- React/TypeScript dashboard with Overview, Data Room, Research Workspace, Retrieval Lab and Evaluation Studio views — the Data Room renders the **original PDF page side by side with the extracted structure and draws each element's real bounding box over it** (`react-pdf`, lazily loaded; `GET /source-versions/{id}/pages` supplies the page box so points map to pixels), a citation opens the exact page with its box highlighted, the Retrieval Lab **runs one query across several databases and profiles at once** and compares stage-by-stage rank movement, and the Evaluation Studio drills into per-question failure causes, diffs manifests and exports CSV/JSON/Markdown;
- **all six design §15.6 interface states designed explicitly** and each reachable in a Playwright test — unavailable databases disabled with a concrete health reason, resumable interrupted jobs, review-required parses, abstention, citation-resolves-or-verifier-defect, and progress/empty/error/partial-success/degraded-model;
- deterministic Success@k, Recall@k, reciprocal-rank, nDCG, abstention-accuracy and repeated/median p95 latency metrics, **cross-checked against `ir-measures`** (the independent reference implementation) to within 1e-6 — proves the hand-rolled formulas are correct, not merely self-consistent;
- a reproducible, fixture-backed **15-question golden set** across all seven design classes (direct-fact, numerical, exact-identifier, multi-hop, metadata-filter, conflicting, unanswerable), passing the three currently implemented gates — see "Golden-set benchmark flow" below;
- **topic-level conflicting-evidence detection** (`conflicts.py`) — flags two citations reporting different numeric values for genuinely related content, replacing a stub that always returned `false`;
- **six of seven vector-store adapters genuinely working, live-verified** — FAISS, Chroma, Qdrant, Milvus and Weaviate all run locally (the first four embedded, no service required), pgvector via the included Docker profile; only Pinecone needs an API key. See "Vector stores" below;
- operator-reviewed golden datasets, reproducibility manifests, synchronous persisted experiment records, SSE completion events and CSV/JSON/Markdown report export;
- SSRF-bounded URL ingestion and source/artifact provenance detail endpoints;
- **capability endpoints that report what is actually reachable, not what is merely installed** (`GET /parsers` adds the formats each parser is routed for; `GET /parsers/routing` publishes the format matrix, the escalation policy and its bake-off-derived thresholds; `GET /vector-stores` reports the design §10 portable contract per store, including the three operations this codebase does *not* implement and whether a store filters in the database or in Python);
- `GET /logical-sources`, closing the gap where nothing exposed design §5.1's one-logical-source-to-many-immutable-versions relationship — `/sources` and `/source-versions` both returned the same flat list — plus `logical_source_id`/`status`/`queryable` filters on both and `GET /golden-datasets/{id}`;
- **a nonce-delimited untrusted-evidence prompt boundary** (design §12/§18) that closed a live-measured prompt-injection hole; see "Security and the trust boundary" below.

## Generation model

The default is **`granite4.2:3b`** (IBM Granite 4.2, Apache-2.0, 2.2 GB on disk /
~2.5 GB VRAM, 128K context). It is trained explicitly for RAG, tool use and structured
JSON output, which is what the citation contract above needs, and it leaves roughly
5.7 GB of an 8 GB card free for the embedding and reranking models.

Granite ships with thinking **on** by default; `INVESTRAG_OLLAMA_DISABLE_THINKING`
(default `true`) sends `think: false` so answers are direct rather than wrapped in a
`<think>` block. Non-thinking models ignore the flag, so it is safe to leave on.

Measured on this machine, 2026-08-29, two passes over 7 questions plus 9 repeat
latency samples:

| | granite4.2:3b |
|---|---|
| Grounded answers with resolvable citations | 10/10 |
| Correct abstentions on off-corpus questions | 4/4 |
| Citation-repair retries fired | 1/10 (rescued the answer) |
| `<think>` leakage into answers | 0/14 |
| Warm generation latency | 2.6 s simple / 3.2 s multi-part / ~6 s summary |

For comparison, llama3.1:8b previously failed the two-part question
"What was revenue and what happened to operating margin?" by omitting its citation
labels entirely.

When a completion is otherwise sound but omits its citation labels — common for small
local models on multi-part questions — one targeted repair retry is issued with the
permitted labels enumerated. Only a second failure withholds the answer, and that case
is reported as a citation failure rather than as missing evidence
(`answer_withheld`, `citation_repair_attempted`, `citation_repair_succeeded`).

## Vector stores

FAISS, Chroma, Qdrant and Milvus all run **embedded** — no service, no Docker. Weaviate
needs Docker (no supported embedded mode on Windows); pgvector needs the included
Postgres profile; Pinecone needs an API key. **Six of these seven adapters were
verified live on this machine, 2026-08-30** — each round-trips a real chunk with
provenance and a `source_id` filter through a real running instance. Only Pinecone
is implemented against documented API but not exercised against a live index.

Milvus specifically required a real fix, not just a flag: `pymilvus`'s own package
metadata excludes `win32` from its `milvus_lite` extra (a stale marker — the actual
`milvus-lite` wheel has supported Windows since 3.2.1). This project depends on
`milvus-lite` directly instead of through that broken extra.

```powershell
cd path\to\investrag-studio
docker compose up -d pgvector
$env:INVESTRAG_POSTGRES_DSN = "postgresql://investrag:investrag@127.0.0.1:5433/investrag"

docker compose --profile weaviate up -d weaviate
$env:INVESTRAG_WEAVIATE_HOST = "127.0.0.1"   # ports 8090/50052 by default, set to avoid clashing with other local Docker projects
```

pgvector's adapter creates the `vector` extension and a provenance-bearing table on
first publication. Weaviate's adapter uses self-provided vectors — this project
always computes its own embeddings, never Weaviate's built-in vectorizers. Neither
is required for the laptop-only path; `GET /api/v1/collections` reports live
per-store count vs. the catalog's expected published-chunk count for whichever
stores are actually configured.

## Retrieval pipeline

The hybrid profile is `dense 25 + BM25 25 → weighted RRF → top 20 → rerank → top k`
(design §11). The reranker is **`mixedbread-ai/mxbai-rerank-base-v2`**, chosen over
the design's original `BAAI/bge-reranker-v2-m3` (absent from current reranker
trending on Hugging Face) after live-comparing it against
`nvidia/llama-nemotron-rerank-1b-v2` and `jinaai/jina-reranker-v2-base-multilingual`:
both require `trust_remote_code=True` (executing a vendor's custom modeling code),
and the Jina model is non-commercially licensed. mxbai-rerank-base-v2 is Apache-2.0,
loads through the standard `sentence_transformers.CrossEncoder` API, and runs on
CPU — measured on this machine: 18.5s cold load (once per process), 236ms warm per
query, correctly ranking a margin-specific passage above unrelated ones for a
margin question.

**Why the composition retrievers aren't imported from LangChain.** The design
originally specified `EnsembleRetriever`, `MultiQueryRetriever`,
`ParentDocumentRetriever` and `SelfQueryRetriever`. None of these exist in current
LangChain (1.3.x) or its docs — they were moved to `langchain-classic`, an
explicitly maintenance-mode compatibility package, and `langchain` 1.x now hard-
depends on `langgraph`. Building the retrieval showcase on a deprecated package
would demonstrate the opposite of current LangChain fluency, so this project
reimplements the same *behaviour* without the deprecated import: the hand-rolled
weighted RRF fusion (`retrieval.py`) implements what `EnsembleRetriever` wraps; the
`parent` profile expands to a real, full-section parent chunk exactly as
`ParentDocumentRetriever` would; multi-query expansion calls `granite4.2:3b`
directly for LLM-generated variants, functionally equivalent to
`MultiQueryRetriever`, with a deterministic fallback. `SelfQueryRetriever` is cut
from scope entirely — it was lab-only in the original plan, and hand-rolling a
natural-language-to-filter parser for a lab-only feature was judged not worth the
cost once no non-deprecated LangChain implementation was available.

Although `langchain` 1.x installs `langgraph` as a dependency, InvestRAG itself stays
a request/response 2-Step RAG pipeline (retrieve-then-generate) and does not define
a stateful agent graph.

## CLI ingestion

```powershell
cd path\to\investrag-studio\backend
uv run python scripts/create_demo_corpus.py ..\.data\demo-corpus
uv run python scripts/seed_demo.py .\path\to\report.pdf .\path\to\metrics.xlsx
uv run python scripts/demo_smoke.py
```

The generated demo corpus includes native and scanned PDFs, XLSX, valid Office-openable DOCX/PPTX packages, HTML, JSON, CSV and an EML with a Markdown attachment. Outlook MSG uses the dedicated `extract-msg` parser. OCR is attempted when the local Tesseract executable is installed; OCR-derived or unavailable text remains visibly partial and is not published for querying without review.

`demo_smoke.py` runs the complete offline smoke path—mixed-format ingestion, recursive attachment handling, FAISS/Chroma/Qdrant publication, a **synthetic pipeline fixture**, retrieval metrics and report persistence—using deterministic hash embeddings and no Ollama/API key. Its generated labels are not presented as a human-reviewed benchmark.

Raw artifacts, the SQLite catalog and the FAISS index are stored under the configured
`INVESTRAG_DATA_DIR` (default: a local `.data/investrag` directory).

## Parser bake-off

```powershell
cd path\to\investrag-studio\backend
uv sync --extra parsers
uv run python scripts/parser_bakeoff.py
```

Runs the demo corpus plus a real bordered financial table and a deliberately malformed
PDF through the real parser stack and writes `docs/reports/parser-bakeoff.md` at the
repository root. The escalation thresholds in `escalation.py` are derived from this
report, not guessed — the run on 2026-08-30 genuinely triggered escalation on two
fixtures, including a case where Docling's internal OCR silently misread "42 percent"
as "42 2 perce" while still reporting `confidence=1.0` on the element; only the
cross-parser disagreement signal caught it. See the report for the full finding and
the OCR-confidence limitation it documents.

## Test

```powershell
cd path\to\investrag-studio\backend
uv run ruff check src tests
uv run mypy src scripts
uv run pytest                    # default run: no network, no GPU, no database

cd ..\frontend
npm run lint
npm run test -- --run
npm run build
npm run test:e2e                 # Playwright: the design §15.6 interface states
```

### End-to-end suite

`npm run test:e2e` needs no servers running. `e2e/global-setup.ts` seeds a real corpus
through the real ingestion pipeline (`backend/scripts/seed_e2e_corpus.py` — a 40-page
PDF, a header-only email that genuinely parses as `partial`, and a file that sniffs as a
PDF and is not one), then starts the real backend on port 8010 over an isolated data
directory; Playwright starts Vite on 5175 against it.

The backend is *not* a Playwright `webServer` entry, deliberately: those start before
`globalSetup`, so the API process would hold the SQLite catalog and FAISS index open
while a second process seeded the same directory. That produced real corruption during
development — `GET /collections` correctly reported FAISS holding 4 vectors against a
catalog of 77 chunks — so seeding now runs to completion in a process that has fully
exited before the API starts.

Eleven of the sixteen tests reach their state through the backend genuinely producing
it. Three shape the `POST /queries` response with `page.route`, each saying so at its
own site: the **verifier-defect** state needs a model to invent a citation label, the
**degraded-model** state needs an Ollama outage *with* evidence to fall back to (the
e2e backend already has an unreachable Ollama, so a live query abstains before
generation), and the **conflict-disclosure** state has no contradiction fixture in the
seeded corpus. All three payloads are produced for real by backend code that has its own
backend tests; the e2e test pins the interface contract.

Screenshot evidence captured from the real running stack lives in
`docs/evidence/phase-8/` and is regenerated with
`npx playwright test e2e/evidence.spec.ts`.

### Test markers

`pyproject.toml` carries `addopts = "-q -m 'not integration'"`, so the default
invocation above is genuinely service-free — which is what makes design §20's last
paragraph ("continuous tests do not require network access, a GPU, or every database")
a checkable claim rather than an intention.

| Marker | In the default run? | What it needs | How to run it |
|---|---|---|---|
| *(unmarked)* | yes | nothing | `uv run pytest` |
| `security` | **yes** — offline and deterministic | nothing | `uv run pytest -m security` |
| `integration` | **no** — deselected by `addopts` | live pgvector / Weaviate services; embedded Chroma, Qdrant, Milvus | `uv run pytest -m integration` |

`-m integration` covers each vector database **through the whole service** — a real
`ingest` parses, chunks, embeds and publishes, and a real `query` retrieves back out —
which is what distinguishes it from `test_vector_adapters.py`'s adapter-level round
trips. Exactly one adapter is open at a time and each test cleans up its own run-unique
collection or table, so Milvus and Weaviate are never resident together. **Never run
that marker under `pytest-xdist`:** parallel workers would defeat the
one-store-at-a-time property the marker exists to preserve.

The two PostgreSQL catalog tests and the two service-backed adapter round trips
previously ran (and skipped cleanly) inside the default suite; they now carry the
`integration` marker, so the default run touches no service at all rather than
attempting a connection and skipping.

This release deliberately does not claim production-scale throughput, cloud deployment or fully active six-database benchmarking. The local three-store path is measured on the laptop; service/cloud adapters are opt-in follow-up work.

## Security and the trust boundary

Design §2.10, §12 and §18 all say the same thing three ways: retrieved documents are
untrusted evidence, they are delimited and labelled as such in the prompt, and a cited
span must support the claim attached to it.

Until 2026-08-30 this codebase satisfied only the *labelling* half. Excerpts were
concatenated into the prompt as `[S1] text` with **no boundary at all**, so a document
body containing the literal string `[S2] …` was byte-for-byte indistinguishable from an
evidence block the retriever had supplied. Measured live against `granite4.2:3b`, 3 runs
of 3, that gap was exploitable: a document that forged an `[S2]` block made the model
answer **$999M** and attribute it to `[S2]`, where no such figure exists. The citation
validator passed it, because `S2` *is* a known label.

`llm.py::build_evidence_prompt` now wraps every excerpt in a region delimited by a
**per-request random nonce**:

```
BEGIN UNTRUSTED DOCUMENT S1 <16-hex nonce>
…document text, with the nonce and any bracketed [Sn] labels neutralised…
END UNTRUSTED DOCUMENT S1 <16-hex nonce>
```

A document cannot close its own region or open a forged one without predicting the
nonce, and `[Sn]` inside a document body is rewritten to `<Sn>` so it cannot impersonate
a label. Re-measured on the same payload, 3 runs of 3: the model now correctly
attributes the correction to `S1`, the region that actually contains it — the
cross-label forgery is gone. Clean-evidence citation compliance was re-checked at the
same time and is 3/3 with both labels.

**What this does not fix, stated plainly.** The model still *believes* an in-document
"authoritative correction". That is content-level trust, not prompt-structure trust, and
the design's answer to it is the conflicting-evidence path — which compares *across*
citations and therefore cannot see a contradiction contained inside a single chunk.

Behind the boundary sits the defence in depth that already existed: an injected
instruction the model does obey produces output citing nothing, and the citation
validator withholds it (`answer_withheld`) and returns the retrieved evidence for the
operator to read instead.

Adversarial coverage lives in `tests/test_security_injection.py` (an 8-payload injection
corpus: role override, forged known/unknown labels, forged question, boundary escape,
instruction in a table cell, exfiltration request, tool-call injection) and
`tests/test_security_fuzzing.py` (truncated and corrupt PDFs, DOCX with a corrupt part,
XXE via an external entity, format confusion where the extension lies about the content,
archive traversal/symlink/file-count/expansion-bomb limits, zero-byte files, hostile
filenames, non-UTF-8 text and deeply nested JSON) — all driven through the **real**
ingestion path rather than against parsers in isolation.

## Architecture and reference matrices

`docs/architecture/investrag-studio.md` (repository root) carries design §23's
documentation set: ports-and-adapters and ingestion/query sequence diagrams in Mermaid,
the parser routing and format-support matrix with the bake-off-derived escalation
thresholds, the vector-store capability matrix (including the three §10 contract
operations this codebase does *not* implement), the experiment-manifest schema and
benchmark methodology, local deployment and resource-profile instructions, corpus
acquisition commands, and the honest known-limitations list.

The parser and vector-store matrices are also served live — `GET /api/v1/parsers`,
`GET /api/v1/parsers/routing` and `GET /api/v1/vector-stores` — and
`tests/test_capabilities.py` asserts the published matrices against what the router and
the adapters actually do, so a matrix cannot drift into aspiration.

## Golden-set benchmark flow

1. Ingest and inspect sources in the Data Room.
2. Collect human-reviewed `chunk_id` or `element_id` judgments from `GET /api/v1/sources/{source_id}`.
3. Save a `GoldenDataset` with `POST /api/v1/golden-datasets`.
4. Run the portable track with `POST /api/v1/experiments` using `faiss`, `chroma` and `qdrant`.
5. Export the measured record from `/api/v1/experiments/{id}/report?format=markdown|csv|json`.

The frontend's Evaluation Studio exposes the same flow without inventing fixture scores. Retrieval Lab also remains empty until a real query emits stage diagnostics. Empty, partial and unavailable states remain visible until measured data exists.

### Real golden set (`scripts/build_golden_set.py`)

```powershell
cd path\to\investrag-studio\backend
uv sync --extra eval
uv run python scripts/build_golden_set.py
```

15 fixture-backed questions across all seven design §14.1 classes. This is a reproducible
engineering benchmark, not a substitute for the approximately 40-question human-verified
set described in the design. `relevant_chunk_ids` are resolved at run time against real ingested chunk
ids, never hardcoded. Writes `docs/reports/golden-set.json` (the dataset) and
`docs/reports/golden-set-gate-report.md` (per-question results, the gate verdict,
and the `ir-measures` cross-check). Current result: **all three provisional
gates pass** — Success@5 1.0, nDCG@5 0.848, abstention accuracy 1.0.

Building this set surfaced two real bugs in the core retrieval/evaluation path,
both fixed with regression tests, not just documented:

1. **Evidence-gate false positive.** A single coincidental stopword-filtered term
   match (the word "current" in a loan's "Current" repayment status matching
   "**current** price of gold") was enough to make the gate bless every retrieved
   candidate for a completely unrelated question. The gate now requires at least
   two overlapping terms for multi-term queries.
2. **Metric aggregation silently punished correct abstention.** An unanswerable
   question has no `relevant_chunk_ids` by construction, so Success@k/Recall@k/
   nDCG@k are mathematically 0 for it regardless of retrieval quality — averaging
   it into the aggregate made a perfectly correct 13/13-answerable run read as
   "86.7% success" before this fix. These metrics are now computed only over
   answerable questions; abstention accuracy remains the separate, correct metric
   for the unanswerable class (design §14.7 keeps them as separate gates for
   exactly this reason).

A regression, while diagnosing the first fix, also caught a third, smaller bug: a
sentence-ending period gets glued onto the preceding word by the tokenizer's
decimal-number regex ("quarter." ≠ "quarter"), invisible while a single-term match
was enough to pass the gate, real once two terms were required.

## Acceptance exercise

All eight design §21 steps were attempted live against a real running stack
(real Ollama, real BGE-M3, real Docker-backed pgvector/Weaviate) on 2026-08-30 — full
write-up in `docs/reports/acceptance-demonstration-2026-08-30.md`, real exported evidence
in `docs/evidence/acceptance-demo/`. The run found and fixed a real generation-reliability
bug (`granite4.2:3b` falling into an unbounded token-repetition loop that defeated the
90 s answer timeout, fixed with `repeat_penalty`/`num_predict`, proven via revert) and
disclosed incomplete acceptance items rather than hiding them: citation-label compliance
under the Phase 9 prompt-injection hardening is not fully reliable on this 3B model (the
verifier-defect safety net caught every failure — no miscited answer was ever shown), and
Retrieval Lab's measured "latency" currently includes a full generation attempt per store
rather than the retrieval-only comparison design §15.4 describes.

## Defensible résumé language

Written only after the gate report and the acceptance demonstration above both existed,
per design §2.12/§23 — every figure below traces to a specific, reproducible run:

> Built InvestRAG Studio, a local-first RAG evaluation platform spanning six vector-store
> backends (FAISS, Chroma, Qdrant, Milvus, Weaviate, pgvector), a Docling/OCR ingestion
> pipeline with measured parser-escalation detection, and a hybrid dense+BM25+RRF retrieval
> stack with cross-encoder reranking — achieving Success@5 1.0 and nDCG@5 0.848 on a
> 15-question fixture-backed golden set, cross-validated against the independent `ir-measures`
> library to within 1e-6. Found and fixed a live prompt-injection vulnerability (forged
> citation labels bypassing the evidence-trust boundary) and a generation-reliability bug
> (unbounded token repetition defeating the answer timeout), each with measured
> before/after proof. Covered by 224 collected backend tests, a 21-test Playwright e2e
> suite, and a documented live acceptance exercise with limitations disclosed.

## Explicit follow-up scope

This is a tested local vertical slice, not the entire architecture-design backlog. The `parent` profile expands to a real full-section parent chunk and `multi-query` calls `granite4.2:3b` for real variant generation, each falling back to deterministic behaviour and reporting which happened in the trace — see "Retrieval pipeline" above for why these are hand-rolled rather than imported from LangChain. `SelfQueryRetriever` is an explicit non-goal, not a deferral. Parent/table-role chunks are not yet filtered out of the direct dense/BM25 search candidate pool — both parent and child (or prose and table-row) chunks remain searchable together, short of design §9.3's "only children are searched" intent. Native/ablation experiment tracks (`ExperimentManifest.track` accepts `"native"`/`"ablation"` but `ExperimentRunner` only implements `"portable"`) and a RAGAS local-judge track remain follow-up work. Ingestion now streams real stage-by-stage progress over SSE (`GET /ingestions/{job_id}/events`); `POST /experiments` is the one remaining long-running endpoint that still blocks synchronously — the job registry it needs already exists (`jobs.py`), the hook-up just has not been done yet. Conflicting-evidence detection is real but topic-level (shared vocabulary + differing numbers with the same unit), not claim-level entailment — it cannot distinguish a genuine contradiction from two numbers about related-but-distinct facts. The current validator resolves citation labels and abstains when the deterministic evidence gate fails; it does not claim claim-level factual entailment.

MinerU escalation has a real, tested port and preflight (`mineru_port.py`), but no MinerU worker is actually provisioned — a real `pip install --dry-run "mineru[core]"` against this project's environment showed it needs `starlette==0.52.1` against FastAPI's `starlette>=1.6.0` plus a conflicting `transformers` pin, confirming it must run in its own venv. Escalation is currently *detected and recorded* (see the parser bake-off report) but not *acted on*. See the architecture document's known-limitations section for the full remaining scope.
