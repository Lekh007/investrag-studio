# InvestRAG Studio — System Design

**Date:** 2026-08-29
**Status:** Approved for implementation
**Target repository:** standalone `investrag-studio` repository
**Primary role alignment:** Vichara GenAI Lead Engineer (Investment Data Platforms)

---

## 1. Purpose

InvestRAG Studio is a local-first investment-document intelligence and retrieval evaluation platform. It demonstrates broad document ingestion, structure-aware parsing, LangChain retrieval patterns, local LLM inference, several vector databases, reproducible RAG evaluation, and an evidence-first React interface.

The product must answer research questions with traceable evidence and expose how an answer was produced. It is not a generic “chat with PDF” demo and it is not a claim of production-scale vector-database performance.

The project demonstrates extensive LangChain usage, heterogeneous document processing, portable and database-native retrieval, reranking, provenance, and controlled evaluation.

## 2. Locked constraints

1. All mandatory functionality must run locally without a paid service.
2. Ollama serves the generation model on the laptop's RTX 4060 GPU with 8 GB VRAM.
3. The default generation model is the existing `llama3.1:8b` quantized model unless measured quality requires a compatible replacement.
4. BGE-M3 produces 1,024-dimensional dense embeddings locally.
5. The local vector-database comparison includes FAISS, Chroma, Qdrant, pgvector, Weaviate, and Milvus. Pinecone is an optional later adapter and is not required for the local release.
6. Resource-heavy databases are run and benchmarked sequentially. The design does not assume that Ollama, MinerU, Milvus, and Weaviate can all consume peak resources simultaneously.
7. Large source artifacts, model caches, and benchmark artifacts live outside Git in the configured local data directory. Docker-managed database data uses named volumes rather than depending on Windows bind-mount performance.
8. The backend uses LangChain Runnables and retrievers. It does not define a LangGraph agent workflow.
9. Every answer must either cite retrievable source evidence or explicitly abstain.
10. Retrieved documents are untrusted evidence, never instructions. Prompt-like text inside documents cannot change system behavior.
11. Claims stay honest. Laptop benchmarks demonstrate controlled local behavior, not enterprise throughput or multi-node scalability.
12. No résumé metric is published until it is produced by a reproducible benchmark run.

## 3. Hardware and runtime assumptions

The target machine was verified with:

- NVIDIA RTX 4060 Laptop GPU, 8,188 MiB VRAM;
- approximately 31 GB system RAM;
- AMD Ryzen AI 9 HX 370, 12 cores;
- approximately 983 GB free on `D:`;
- Docker Engine 28.5.2;
- Ollama listening locally at `127.0.0.1:11434`;
- `llama3.1:8b` installed with completion and tool support.

The generation model is not a vision model. OCR and layout extraction therefore remain deterministic parser responsibilities rather than being delegated to the chat model.

## 4. Product architecture

```text
React + TypeScript web application
├── Overview
├── Data Room
├── Research Workspace
├── Retrieval Lab
└── Evaluation Studio
                 │ HTTP + server-sent events
                 ▼
FastAPI application
├── source/version service
├── ingestion coordinator
├── parser router and quality gates
├── canonical document store
├── embedding service
├── vector-store adapters
├── LangChain retrieval pipelines
├── answer generator and citation verifier
└── experiment/evaluation service
          │
          ├── PostgreSQL metadata and experiment records
          ├── immutable raw/derived artifacts on D:
          ├── local Ollama model gateway
          └── vector stores
              ├── FAISS
              ├── Chroma
              ├── Qdrant
              ├── pgvector
              ├── Weaviate
              └── Milvus
```

The backend follows ports and adapters. Domain records, quality policies, retrieval configurations, citations, and experiment definitions do not import a database-specific SDK. Each parser and vector store implements an adapter contract. Tests replace local services with deterministic fakes.

The first release binds to localhost and assumes a single local operator. Authentication and remote exposure are outside the MVP. A future remote deployment must add authentication, authorization, TLS, secret management, and tenant isolation before it is safe to use.

## 5. Core data model and provenance

### 5.1 Source and version records

A logical source can have several immutable versions. Each version records:

- source identifier and version identifier;
- original filename, URI, or API endpoint;
- media type and detected format;
- retrieval or upload timestamp;
- content checksum and byte size;
- parent source when extracted from an archive or email attachment;
- parser, OCR engine, model, and configuration versions;
- ingestion status and quality flags;
- raw and derived artifact locations.

The same checksum is not processed twice under the same parser configuration unless the operator explicitly requests it.

### 5.2 Canonical elements

Parsers emit canonical elements rather than unstructured strings. An element includes:

- stable element identifier;
- source and version identifier;
- element type such as heading, paragraph, list, table, table row, figure, caption, spreadsheet range, code block, or email field;
- normalized text and optional structured payload;
- hierarchy path and reading order;
- page, slide, sheet, cell range, row range, JSONPath, or character span;
- bounding box where available;
- parser confidence and warnings;
- parent and child element identifiers.

Stable source-element identifiers form the evidence ground truth. Chunk identifiers are derived and may change when chunking experiments change.

### 5.3 Chunks and embeddings

Every chunk records:

- chunk profile and version;
- ordered canonical element identifiers;
- parent document or section identifier;
- text used for embedding;
- token count;
- inherited metadata and security flags;
- embedding model, dimensions, normalization policy, and content checksum.

Embeddings are cached by normalized chunk checksum plus embedding-model version. One computed embedding set is reused across database adapters during the portable benchmark.

## 6. Supported input formats

The ingestion surface supports:

- native and scanned PDF;
- DOCX;
- PPTX;
- XLSX and CSV;
- HTML;
- JSON and API responses;
- Markdown and plain text;
- PNG, JPEG, and TIFF images;
- EML and MSG email, including recursively processed attachments;
- ZIP archives with safe extraction limits.

Archive extraction rejects absolute paths, parent traversal, symlinks, excessive nesting, excessive file counts, and uncompressed-size expansion beyond configured limits. Password-protected or unsupported items become visible partial failures rather than being silently skipped.

## 7. Parser strategy

No parser is treated as universally best. InvestRAG uses deterministic format routing followed by measured escalation.

### 7.1 Primary and specialist parsers

- **Docling** is the primary layout-aware parser for PDF, DOCX, PPTX, HTML, and image-oriented documents. PDF uses accurate table structure mode where practical, OCR when required, and element-level provenance.
- **MinerU 2.5** is an escalation parser for difficult scans, formulas, dense tables, multi-column reading order, or low-confidence Docling output. It runs as a separate resource profile while the Ollama generation model is unloaded when GPU memory requires it.
- **PyMuPDF4LLM** is a fast second opinion for native PDFs and a validation path for page text and reading order.
- **openpyxl** is authoritative for workbook structure, formulas, cell coordinates, merged ranges, hidden sheets, and sheet metadata.
- **Python-native JSON, CSV, Markdown, MIME, and archive libraries** preserve their formats' natural structure instead of passing everything through a PDF-oriented parser.

Marker is not a mandatory parser because its model licensing and local service constraints make it a weaker portfolio default. It can be considered later only after an explicit license and resource review.

### 7.2 Escalation policy

Docling output is accepted when required pages are present, reading order passes validation, tables are structurally consistent, OCR confidence is sufficient, and citations can be reconstructed. MinerU is requested when any of the following apply:

- scanned or mixed-image PDF with low OCR confidence;
- missing or collapsed tables;
- broken multi-column reading order;
- formula-heavy content;
- large disagreement with the PyMuPDF validation path;
- quality score below the publication threshold.

If both attempts fail, the version remains searchable only when safe partial content is clearly marked. It never appears as a fully successful ingestion.

### 7.3 Parser bake-off

Before corpus ingestion, a small adversarial test pack compares parsers on native prose, scans, multi-column research reports, financial tables, formulas, slides, spreadsheets, email attachments, and malformed files. Human-reviewed expected elements and locations determine the routing thresholds. Vendor benchmark claims inform parser selection but do not replace this local test.

## 8. Ingestion and document-processing flow

1. Accept a file, URL, API response, email, or archive.
2. Stream it to an immutable staging artifact while calculating checksum and size.
3. Detect the actual format rather than trusting the extension.
4. Run security and resource-limit checks.
5. Create the source version and processing manifest.
6. Route to the format-native parser.
7. Normalize output into canonical elements with provenance.
8. Run quality gates and optional parser escalation.
9. Persist raw output, normalized elements, warnings, and quality scores.
10. Create one or more versioned chunk profiles.
11. Generate or reuse BGE-M3 embeddings.
12. Publish the same embedding set through selected vector-store adapters.
13. Run index consistency checks and make the version queryable.
14. Emit stage progress and errors to the UI through server-sent events.

The process is idempotent and checkpointed. A failed vector-store publication does not require reparsing or re-embedding the document. Restarting resumes at the first incomplete stage.

## 9. Chunking profiles

InvestRAG provides four controlled profiles:

1. **Structure-aware default:** Docling HybridChunker respects headings, lists, tables, captions, and tokenizer limits.
2. **Recursive baseline:** LangChain RecursiveCharacterTextSplitter provides a conventional comparison.
3. **Parent-child:** small child chunks are embedded, while larger parent sections are returned as final context.
4. **Table-aware:** tables are indexed with titles, headers, row groups, units, and nearby narrative context. Large tables receive both summary-level and row-group representations.

Chunk profiles are immutable experiment inputs. Re-chunking creates a new profile version and never silently replaces an earlier benchmark configuration.

## 10. Vector-store adapter contract

Every adapter supports a common portable contract:

- create or identify a versioned collection;
- upsert dense vectors with canonical metadata;
- delete an indexed source version;
- dense similarity search;
- metadata filtering;
- maximum marginal relevance where available or implemented portably;
- health and capability reporting;
- collection statistics and disk-size reporting.

Database-native sparse or hybrid features live behind explicit capabilities and never alter the portable benchmark silently.

### Database roles

- **FAISS:** lightweight in-process dense and MMR baseline.
- **Chroma:** developer-friendly persistent dense/filter baseline with external BM25.
- **Qdrant:** native dense and sparse vectors with reciprocal-rank fusion.
- **pgvector:** dense search combined with PostgreSQL full-text search and fusion.
- **Weaviate:** vector search plus native BM25F/hybrid behavior.
- **Milvus:** dense retrieval plus built-in sparse/BM25 functionality.
- **Pinecone:** optional cloud adapter added later without changing domain or experiment schemas.

## 11. LangChain retrieval design

Queries pass through an explicit Runnable pipeline whose stages are visible in the trace:

1. normalize the question;
2. optionally rewrite it while preserving identifiers, dates, units, and quoted phrases;
3. derive and validate metadata filters against an allowlist;
4. retrieve candidates;
5. fuse dense and lexical rankings when hybrid mode is active;
6. rerank candidates;
7. expand parents or neighboring table context;
8. build a token-bounded evidence packet;
9. generate a structured answer;
10. validate citations and deterministic facts;
11. attempt one bounded repair or abstain.

### Retrieval profiles

- dense similarity;
- MMR;
- portable hybrid using a vector retriever, BM25, and LangChain `EnsembleRetriever` with weighted reciprocal-rank fusion;
- `ParentDocumentRetriever`;
- `MultiQueryRetriever` for ambiguous research questions;
- `SelfQueryRetriever` in the lab only because generated filters require strict validation;
- contextual compression with local `BAAI/bge-reranker-v2-m3`.

The initial production profile is:

```text
dense top 25 + BM25 top 25
        │
        ▼
weighted reciprocal-rank fusion → top 20
        │
        ▼
BGE reranker → top 6–8
        │
        ▼
parent/table expansion → evidence packet
```

All counts are configurable experiment inputs, not hidden constants.

## 12. Answer and citation policy

The generator returns a typed record containing:

- answer text;
- citation references;
- `insufficient_evidence` flag;
- `conflicting_evidence` flag;
- concise explanation of uncertainty.

The context contains explicit source boundaries and labels every excerpt as untrusted. The verifier checks:

- every citation resolves to an indexed source element;
- cited spans support the associated claim;
- numerical claims appear in evidence or deterministic extracted fields;
- dates and units are preserved;
- conflicts are disclosed;
- an unanswerable question produces abstention.

One repair attempt may replace invalid citations or remove unsupported claims. A second failure returns a controlled refusal with the available evidence, never an unsupported answer.

## 13. Retrieval trace

Every query stores:

- original and rewritten query;
- validated filters;
- vector store and collection version;
- chunk, embedding, retriever, fusion, reranker, prompt, and model versions;
- dense, lexical, fused, and reranked ranks and scores;
- parent expansion decisions;
- final context identifiers and token counts;
- retrieval, reranking, generation, and end-to-end latency;
- generated answer, citations, validator outcomes, and repair decision;
- local hardware/resource sample.

The trace powers debugging and the Retrieval Lab. It is not hidden in application logs.

## 14. Evaluation methodology

One composite “RAG score” is prohibited. Evaluation separates parser quality, retrieval relevance, generation quality, and systems performance.

### 14.1 Golden set

The initial golden set contains approximately 40 manually verified questions:

- 8 direct facts;
- 8 numerical or table questions;
- 6 exact-identifier or keyword-heavy questions;
- 6 cross-document or multi-hop questions;
- 4 metadata-filter questions;
- 4 conflicting or version-aware questions;
- 4 unanswerable or prompt-injection questions.

Each record includes the question class, answerability, expected filters, reference facts, required citations, and source-element judgments graded 0/1/2. Candidate questions may be generated automatically, but a human verifies the evidence before the item becomes benchmark ground truth.

### 14.2 Parser metrics

- textual and numerical token coverage;
- reading-order checks;
- table structural validity;
- OCR confidence;
- provenance reconstructability;
- manual-review rate;
- partial and failed document rate.

### 14.3 Retrieval metrics

The deterministic retrieval report uses `ir-measures` and reports:

- Success@1/3/5/10;
- Recall@5/10/20;
- reciprocal rank or MRR@10;
- nDCG@10;
- metadata-filter accuracy;
- duplicate-result rate;
- table-evidence recall;
- per-query-class breakdown.

Success@k means that at least one relevant item appeared. Recall@k means the fraction of all known relevant evidence recovered. The labels are never used interchangeably.

### 14.4 Generation metrics

Primary deterministic metrics are:

- exact and numerical answer accuracy;
- unit and date preservation;
- citation validity, precision, recall, and coverage;
- unsupported-number rate;
- abstention accuracy;
- conflict disclosure;
- output-schema validity.

Optional RAGAS metrics include faithfulness, response relevance, context precision, context recall, and factual correctness. They run through the local Ollama endpoint and are clearly labelled **local LLM judge scores**. Repeated judge runs report variation; they do not override deterministic or human judgments.

### 14.5 Performance metrics

- total ingestion time split into parsing, OCR, chunking, embedding, and indexing;
- index time excluding cached embedding generation;
- cold query latency;
- warm p50 and p95 retrieval latency;
- reranking and generation latency;
- end-to-end latency;
- peak CPU, RAM, and VRAM;
- index size and disk usage;
- failure and retry counts.

Retrieval latency receives warm-up runs and repeated measurements. Expensive end-to-end generation uses fewer repetitions and reports that sample count. Hardware, dependency versions, models, database configuration, seeds where supported, corpus checksum, and experiment manifest are recorded.

### 14.6 Experiment tracks

**Portable track:** every database receives the same cached BGE-M3 embeddings, metadata, filters, candidate counts, and fusion/reranking stages. This is the apples-to-apples leaderboard.

**Native-feature track:** each database demonstrates its best supported lexical/hybrid capabilities. Results are shown as feature demonstrations and are never mixed into the portable ranking.

**Ablation track:** bounded comparisons isolate chunking, dense versus hybrid retrieval, reranking, parent expansion, and multi-query behavior. The platform avoids an unreviewable Cartesian product of every option.

### 14.7 Provisional gates

Initial engineering targets are:

- Success@5 at least 0.90;
- nDCG@10 at least 0.75;
- citation validity 1.00;
- unsupported-number rate 0;
- answerable-question accuracy at least 0.80;
- abstention accuracy at least 0.90;
- no unexplained regression greater than five percentage points.

These are starting acceptance thresholds. They are adjusted only with a documented reason and do not become résumé claims until an actual report satisfies them.

## 15. React product design

The frontend uses React, TypeScript, TanStack Query, Recharts, and a PDF renderer. Server-sent events stream ingestion and experiment progress. WebSockets are not required by the MVP.

### 15.1 Overview

- FastAPI, Ollama, PostgreSQL, and vector-store health;
- corpus, source-version, and ingestion-quality totals;
- active parser, chunk, embedding, retrieval, and generation profiles;
- recent experiments and provisional gate status;
- local hardware and available resource summary.

### 15.2 Data Room

- upload or import file, URL, API response, email, or archive;
- observe each ingestion stage and recoverable error;
- browse source versions and manifests;
- compare the original page with extracted structure;
- inspect tables, normalized elements, chunks, bounding boxes, and warnings;
- reprocess with an alternative parser or chunk profile;
- prevent low-confidence content from appearing as silently trusted evidence.

### 15.3 Research Workspace

- select corpus, database, and retrieval profile;
- ask a research question;
- read a structured answer with inline citations;
- open an evidence drawer at the exact page, slide, cell, row, JSONPath, or email attachment;
- show conflict and insufficient-evidence banners;
- expand the complete query trace.

### 15.4 Retrieval Lab

- run the same query against selected databases or profiles;
- compare dense, lexical, fused, and reranked stages;
- inspect rank movement, scores, filters, chunks, and parent expansion;
- compare quality and latency without generating an answer;
- save a query as a golden-set candidate.

### 15.5 Evaluation Studio

- select a golden dataset and experiment track;
- launch, cancel, checkpoint, and resume a benchmark;
- visualize nDCG, Recall, MRR, citation quality, p50/p95 latency, accuracy-versus-latency, and resource usage with Recharts;
- drill into individual questions and failure causes;
- compare experiment manifests;
- export CSV, JSON, and Markdown reports.

### 15.6 Required interface states

- unavailable databases are disabled with a concrete health explanation;
- interrupted jobs retain their completed stages and can resume;
- low-confidence parses require review or explicit override;
- missing evidence produces an abstention state;
- citation selection always resolves to visible provenance or displays a verifier defect;
- progress, empty, error, partial-success, and degraded-model states are designed explicitly.

## 16. API surface

The conceptual FastAPI resources are:

- `/sources` and `/source-versions`;
- `/ingestions` and stage event streams;
- `/parsers` and `/chunk-profiles`;
- `/vector-stores` and capability/health endpoints;
- `/collections` and publication status;
- `/queries` and query traces;
- `/golden-datasets`;
- `/experiments` and progress event streams;
- `/artifacts/{id}` for permitted local previews and exports.

HTTP schemas are generated from typed request and response models. Long-running operations return job identifiers rather than holding an HTTP request open.

## 17. Failure handling

- Parser failure creates a visible failed or partial version with retained diagnostics.
- Embedding failure retries bounded transient errors and resumes from cached chunks.
- One database publication failure does not invalidate successful publications to other stores.
- A database outage removes that adapter from selection without crashing the application.
- Ollama outage preserves ingestion, retrieval-only comparison, and existing benchmark browsing while answer generation reports degraded status.
- Corrupt indexes fail checksum/consistency checks before becoming active.
- Benchmark interruption retains completed query results and resumes idempotently.
- Resource preflight prevents known incompatible heavy services from launching together.

## 18. Security and trust boundary

- The first release listens on localhost only.
- URL ingestion blocks localhost, private-network, link-local, and cloud-metadata destinations unless explicitly allowlisted.
- Downloads enforce content, time, redirect, and size limits.
- Filenames never determine output paths directly.
- Archive extraction prevents traversal and decompression bombs.
- HTML is sanitized before rendering.
- Retrieved content is delimited and labelled untrusted in prompts.
- Generated answers cannot execute commands or automatically trigger ingestion.
- Raw source and derived artifacts are addressed by internal identifiers, not arbitrary client-supplied paths.
- Logs and exported traces redact secrets and local absolute paths where exposure is unnecessary.

## 19. Local deployment and resource profiles

Docker Compose supplies PostgreSQL/pgvector and the selected service-backed database. FAISS and Chroma may run in-process or as local persistent libraries. Named Docker volumes hold service state; raw and derived corpus artifacts remain on `D:`.

Resource profiles are explicit:

- **interactive:** Ollama plus one lightweight vector store;
- **parser-heavy:** MinerU active, Ollama unloaded when required;
- **database-benchmark:** one service-backed vector database active at a time;
- **Milvus/Weaviate:** dedicated sequential sessions with other heavy stores stopped.

The UI and experiment manifest record the active profile so constrained local runs are reproducible rather than appearing to be simultaneous production infrastructure.

## 20. Test strategy

- unit tests for canonical records, quality rules, filter validation, fusion, citations, and deterministic metrics;
- parser contract tests using small checked-in fixtures for every supported format;
- security tests for malicious archives, unsafe URLs, HTML, prompt injection, and malformed files;
- vector-adapter contract tests against deterministic embeddings;
- integration tests for each local vector database, tagged so heavy services run sequentially;
- retrieval golden tests with no Ollama dependency;
- generation tests using a deterministic model fake plus optional local-model acceptance tests;
- frontend component tests for evidence, trace, progress, and degraded states;
- end-to-end smoke test covering ingestion, publication, query, citation navigation, benchmark, and report export.

Continuous tests do not require network access, a GPU, or every database. Hardware-dependent acceptance tests are separate and record the environment.

## 21. Acceptance demonstration

The project is portfolio-ready when a reviewer can:

1. ingest a mixed corpus containing native and scanned PDFs, a financial spreadsheet, a slide deck, HTML/JSON, and an email attachment;
2. inspect exact provenance and see a low-confidence document take the configured parser-escalation path;
3. ask a cross-format investment-research question and open every citation at its page, table, slide, cell, or JSON location;
4. ask an unsupported question and receive a correct abstention;
5. compare retrieval stages across at least FAISS, Qdrant, pgvector, and one of Weaviate or Milvus;
6. run the portable benchmark and a native-feature demonstration;
7. inspect failures by query class and export a reproducible report;
8. start the demonstrated configuration from documented local commands with automated tests passing.

The remaining database adapters may be demonstrated sequentially, but every claimed adapter must pass its contract tests before appearing in the README or résumé.

## 22. Explicit non-goals

- claiming two years of GenAI experience from this project;
- claiming production high availability, cloud deployment, GPU-cluster operation, or multi-tenant security;
- fine-tuning a foundation model;
- training a proprietary embedding model;
- presenting local laptop latency as a universal database ranking;
- using Pinecone or any paid/cloud dependency in the mandatory path;
- automatically trusting parser confidence or LLM-as-judge scores;
- building autonomous agents that can mutate data or infrastructure.

## 23. Documentation and portfolio evidence

The completed repository must include:

- architecture and sequence diagrams;
- parser routing and format support matrix;
- vector-store capability matrix;
- experiment manifest schema and benchmark methodology;
- local deployment and resource-profile instructions;
- sample mixed-format corpus or reproducible public-source acquisition script;
- screenshots or a short deterministic demo flow;
- known limitations and honest interpretation guidance;
- measured, reproducible résumé bullets only after acceptance evidence exists.

## 24. User responsibilities

Codex handles implementation, dependency configuration, parser and database integration, tests, benchmark tooling, documentation, and debugging. The user is responsible only for guided non-coding verification:

1. keep the laptop powered and awake during large model downloads, parser runs, and database benchmarks;
2. verify the evidence and expected answer facts for golden-set candidates before they become ground truth;
3. perform final visual acceptance using the supplied demo checklist;
4. approve any Windows or Docker prompt that cannot be completed programmatically;
5. review final README and résumé language before it is published externally.

## 25. External technical references

The design relies on official or primary project documentation, including:

- LangChain retrieval integrations and retrievers: <https://python.langchain.com/docs/integrations/retrievers/>
- Docling documentation: <https://docling-project.github.io/docling/>
- MinerU repository and model documentation: <https://github.com/opendatalab/MinerU>
- PyMuPDF4LLM documentation: <https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/>
- BGE-M3 model documentation: <https://huggingface.co/BAAI/bge-m3>
- `ir-measures` definitions: <https://ir-measur.es/en/latest/measures.html>
- RAGAS metrics: <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>
- LangSmith local evaluation: <https://docs.langchain.com/langsmith/local>
- Recharts examples: <https://recharts.github.io/en-US/examples/>

These references guide implementation. Actual capability claims remain subject to pinned-version contract tests on the target machine.

## 26. Open implementation decisions

The following are intentionally deferred to the implementation plan and small technical spikes; they do not change the approved product design:

- exact pinned dependency versions compatible with Python and Docker on the machine;
- whether Chroma runs embedded or in its service mode;
- exact PostgreSQL schema and migration tool;
- frontend PDF component selection;
- final parser quality thresholds derived from the bake-off;
- final retrieval candidate counts and RRF weights derived from the golden set;
- which of Weaviate or Milvus is included first in the acceptance demonstration;
- whether the existing monorepo receives a new top-level application package or InvestRAG is isolated as a sibling package inside the same portfolio repository.

No implementation begins until the repository layout choice is made in the implementation plan and this consolidated design has been reviewed.
