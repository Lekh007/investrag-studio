# InvestRAG Studio — design §21 acceptance exercise

**Date:** 2026-08-30
**Method:** all eight steps attempted live against a real running stack — real `uvicorn` backend, real Vite frontend, real Ollama (`granite4.2:3b`), real BGE-M3 embeddings, real Docker-backed pgvector and Weaviate, real FAISS/Chroma/Qdrant embedded stores. Evidence artifacts are in `docs/evidence/acceptance-demo/`. Steps 2, 6, and 7 are partial for the reasons documented below.

**Corpus:** `scripts/create_demo_corpus.py`, generated fresh for this run — 10 source files (native PDF, scanned/OCR'd PDF, XLSX, DOCX, PPTX, HTML, JSON, CSV, EML-with-attachment) plus 1 recursively-extracted email attachment, 11 source versions, 14 chunks, published to FAISS/Chroma/Qdrant/pgvector/Weaviate.

---

## Step 1 — ingest a mixed corpus

All 10 files ingested through the real `POST /api/v1/ingestions` job-id/SSE endpoint (Phase 8), each polled to completion. Result: 11/11 source versions `status: ready`, `queryable: true` (10 files + `analyst-note.md`, recursively extracted from `research-note.eml`). `GET /api/v1/collections` confirmed all five active stores report `count: 14, consistent_with_catalog: true`; Milvus correctly shows `count: 0, consistent_with_catalog: null` (reachable, not in this run's publish list — not a failure); Pinecone correctly `reachable: false` (no API key, never required). Raw response saved: `source-versions.json`, `collections.json`.

## Step 2 — provenance and the escalation path

`scanned-report.pdf` (a synthetic image-only PDF) carries a real warning on its source version:

> escalation policy indicated MinerU review (reading-order score 0.50 below 0.55; parser disagreement (jaccard 0.36) below 0.40); MinerU was not invoked for this ingestion — see mineru_port.py for the escalation path

This is a genuine escalation trigger, not staged: Tesseract's real OCR misread the synthetic image's text as *"Scanned PDF: operating margin 42 2 percent"* (a spurious inserted "2"), which is exactly the kind of disagreement the Docling/PyMuPDF4LLM cross-check is designed to catch. Confirmed both via the API response and live in the Data Room's provenance panel (`get_page_text` on the opened detail view showed the same chunk text, bbox, and structure). `requires_review` is `false` because escalation is *detected and recorded, not acted on* (MinerU is not provisioned). This is a disclosed implementation gap: the design requires low-confidence evidence to require review or take the configured escalation path.

## Step 3 — cross-format question, citations across location types

Real citations resolved across every location type design §21 asks for, confirmed across two live queries:

| Label | Source | Location |
|---|---|---|
| metrics.csv | table/plain excerpt |
| brief.md | plain excerpt |
| metrics.xlsx | `Summary · A2:B2` / `A3:B3` — real cell reference |
| native-report.pdf | `Page 1` |
| update.pptx | `Page 1` (slide) |
| scanned-report.pdf | `Page 1` |
| metrics.json | `$.company` — real JSONPath |

**A real bug was found and fixed during this step.** The first three live queries through the running app timed out at exactly the 90 s `httpx` client timeout in `llm.py`, degrading correctly to the extractive-fallback path (proving §17's degradation design works) but never producing a generated answer. Diagnosis: a raw `POST /api/generate` with the identical prompt reproduced it — `granite4.2:3b`, even with `think: false` set, produced 1,392 tokens of near-verbatim repeated sentences (70.8 s) before finally emitting `</think>` and a correct 3-line answer. Adding `repeat_penalty: 1.3` and `num_predict: 800` to `llm.py::_payload` fixed it: the same prompt returned a correct, cited answer in 8.8 s. Fixed in `llm.py`, proven via revert (`tests/test_generation.py::test_generation_options_bound_repetition` fails with `KeyError: 'repeat_penalty'` without the fix, passes with it restored), backend restarted with the fix live.

**Disclosed, not fixed: citation-label compliance.** After the timeout fix, three further live queries all completed generation within the timeout (44.9 s, 51.5 s, 77.2 s) but each failed the citation-label verifier — `granite4.2:3b` answered without properly bracketed `[Sn]` labels, and the citation-repair retry also failed each time. The system correctly withheld every one of these as a **verifier defect** rather than showing an uncited or miscited answer — the safety mechanism worked in all three cases. This is a genuine, reproducible finding: the Phase 9 prompt-injection hardening (nonce boundaries, an extensive trust-rules preamble) may be harder for a 3B model to follow through to a correctly labeled citation than the simpler pre-hardening prompt Phase 1's reliability check used. Root cause not further investigated in this pass; flagged as real follow-up scope, not silently patched or hidden.

## Step 4 — unsupported question, correct abstention

*"What is the CEO's annual salary and home address?"* → `abstained`, `0 context chunks`, `generation: 0 ms`, **no model call was made**. The evidence gate correctly found nothing supporting the question before ever reaching Ollama. (A second, milder case — *"Which documents contain numerical evidence of changing margins?"* — also abstained on this exact corpus phrasing, an honest if unintended second data point for the same gate.)

## Step 5 — retrieval-stage comparison across ≥3 real databases

Retrieval Lab run against **FAISS, Qdrant, Weaviate, pgvector** simultaneously (4 of the 5 published stores; Chroma/Milvus available but not selected for this run). Real stage-by-stage candidate survival (Dense → Lexical → Fused RRF → Reranked → Final → Evidence gate) and per-chunk rank movement, with all four stores agreeing 4/4 on every surviving chunk's final rank — expected and reassuring on a 14-chunk corpus, and itself a live sanity check that all four adapters are storing and returning the same underlying vectors correctly.

**Disclosed finding, not fixed:** design §15.4 states Retrieval Lab compares "quality and latency **without generating an answer**." Measured round-trip times (FAISS 139.7 s, Qdrant 52.6 s, Weaviate 14.7 s, pgvector 135.0 s) were far larger and far more variable than the retrieval-only figures shown alongside them (5.3–5.9 s each), because the Lab reuses the same `POST /api/v1/queries` endpoint used for full generation — it appears to run a full generation attempt per selected store, not a retrieval-only path. The retrieval-stage comparison data itself is real and correct; the "latency" framing includes generation time the design says it shouldn't. Flagged as real follow-up scope for Phase 8/9.

## Step 6 — portable benchmark

No golden dataset existed for this freshly-generated corpus (the Phase 7 golden set targets its own separate fixture corpus), so a real 5-question dataset was authored against this exact corpus's real chunk ids (`golden-dataset.json`) — 4 answerable classes (direct-fact, numerical, multi-hop, exact-identifier) plus 1 deliberately unanswerable question. Run against FAISS/Chroma/Qdrant:

- Success@10 **1.000**, Recall@10 **1.000**, MRR@10 **1.000**, nDCG@10 **0.9887**, abstention accuracy **1.000**, p95 retrieval **163–173 ms** per store.

Full manifest and per-question metrics: `portable-benchmark-report.md`. **Native-feature demonstration is not implemented.** `ExperimentManifest.track` accepts `"native"`/`"ablation"` in the schema but `ExperimentRunner` only implements `"portable"`; acceptance step 6 is therefore partial.

## Step 7 — failure inspection and export

Per-question drill-down showed all 15 question/store runs with cause (`ok` / `Unanswerable question — correctly abstained.`) — no genuine failures to inspect on this hand-authored 5-question set, which is itself honest (a small, carefully matched demo set should score well; this is not evidence a 40-question adversarial set would). CSV and Markdown export both pulled live from the real experiment record, not the UI cache: `portable-benchmark-results.csv`, `portable-benchmark-report.md`.

## Step 8 — clean-checkout startup with automated tests passing

Already established by this session's independent verification pass (separate from this demonstration): `ruff check`, `mypy --strict`, the full default `pytest` suite, the `security` and `integration` marker suites, frontend `tsc`, `vitest`, `vite build`, and a from-scratch 21-test Playwright e2e run all pass from documented commands (`README.md`'s "Run the backend" / "Run the frontend" / "Test" sections). A fresh `uv sync --no-dev --no-default-groups` (light install, no extras) confirmed `PostgresCatalog` imports cleanly, proving `psycopg` is correctly a base dependency.

---

## Summary

The live exercise produced useful evidence and found a real generation-reliability bug, but it is not a complete §21 acceptance pass. Parser escalation was detected rather than executed, the native-feature track was not implemented, and the small fixture-backed dataset produced no genuine retrieval failure to inspect. Citation-label compliance and Retrieval Lab's generation-inclusive latency also remain disclosed follow-up work.
