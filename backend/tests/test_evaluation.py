from investrag.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank, success_at_k


def test_ir_metrics_keep_success_and_recall_distinct() -> None:
    relevant = {"a", "b"}
    retrieved = ["x", "a", "y"]
    assert success_at_k(relevant, retrieved, 3) == 1.0
    assert recall_at_k(relevant, retrieved, 3) == 0.5
    assert reciprocal_rank(relevant, retrieved, 3) == 0.5
    assert ndcg_at_k({"a": 2, "b": 1}, retrieved, 3) > 0


def test_ir_measures_cross_check_agrees_with_the_hand_rolled_metrics(tmp_path) -> None:
    """Real (unmocked) cross-check: run an actual golden-dataset experiment through
    the real service/API stack, then verify ir-measures - the citable, independent
    reference implementation - agrees with this project's own success/recall/rr/nDcg
    to within floating-point tolerance. Proves the hand-rolled metrics are correct,
    not merely self-consistent."""

    import pytest

    pytest.importorskip("ir_measures")

    from fastapi.testclient import TestClient

    from investrag.config import Settings
    from investrag.domain import ExperimentRecord
    from investrag.evaluation import cross_check_with_ir_measures
    from investrag.main import create_app

    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    from tests.ingest_helper import ingest_file

    ingested = ingest_file(client, "brief.md", b"# Outlook\nRevenue is stable. Margin improved.", "text/markdown")
    chunks = client.get(f"/api/v1/sources/{ingested['source_id']}").json()["chunks"]
    dataset = {
        "dataset_id": "irmeasures-smoke",
        "name": "ir-measures cross-check",
        "questions": [
            {"question_id": "q1", "question": "What is the outlook?", "relevant_chunk_ids": [chunks[0]["chunk_id"]]},
            {"question_id": "q2", "question": "What about margin?", "relevant_chunk_ids": [chunks[0]["chunk_id"]]},
            {"question_id": "q3", "question": "What is the gold price?", "answerable": False},
        ],
        "corpus_source_ids": [ingested["source_id"]],
    }
    assert client.post("/api/v1/golden-datasets", json=dataset).status_code == 200
    run = client.post(
        "/api/v1/experiments",
        json={"manifest": {"dataset_id": "irmeasures-smoke", "vector_stores": ["faiss"], "profiles": ["hybrid"], "top_k": 3, "repetitions": 1}},
    )
    assert run.status_code == 200
    record = ExperimentRecord.model_validate(run.json())

    cross_check = cross_check_with_ir_measures(record)
    assert cross_check.available, cross_check.reason
    assert cross_check.agrees, f"max difference {cross_check.max_absolute_difference} exceeds tolerance: {cross_check.reason}"
    assert cross_check.max_absolute_difference < 1e-6


def test_success_and_ndcg_exclude_unanswerable_questions_from_the_average(tmp_path) -> None:
    """Regression guard: an unanswerable question has no relevant_chunk_ids by
    construction, so success_at_k/recall_at_k/ndcg_at_k are always 0 for it -
    blending it into the aggregate used to silently punish a perfectly-correct
    system for correctly abstaining. Found via the golden set, 2026-08-30: 13/13
    correct answerable questions plus 2/2 correct abstentions read as "86.7%
    success" before this fix, purely from averaging in the structural zeros."""

    from fastapi.testclient import TestClient

    from investrag.config import Settings
    from investrag.main import create_app

    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    from tests.ingest_helper import ingest_file

    ingested = ingest_file(client, "brief.md", b"# Outlook\nRevenue grew this quarter.", "text/markdown")
    chunks = client.get(f"/api/v1/sources/{ingested['source_id']}").json()["chunks"]
    dataset = {
        "dataset_id": "unanswerable-weighting-smoke",
        "name": "unanswerable weighting smoke",
        "questions": [
            {"question_id": "answerable", "question": "What happened to revenue this quarter?", "relevant_chunk_ids": [chunks[0]["chunk_id"]]},
            {"question_id": "unanswerable", "question": "What is the price of gold?", "answerable": False},
        ],
        "corpus_source_ids": [ingested["source_id"]],
    }
    assert client.post("/api/v1/golden-datasets", json=dataset).status_code == 200
    run = client.post(
        "/api/v1/experiments",
        json={"manifest": {"dataset_id": "unanswerable-weighting-smoke", "vector_stores": ["faiss"], "profiles": ["hybrid"], "top_k": 3, "repetitions": 1}},
    )
    assert run.status_code == 200
    metrics = {m["name"]: m["value"] for m in run.json()["metrics"]}
    assert metrics["faiss/hybrid/success@3"] == 1.0, "a single correctly-retrieved answerable question must not be dragged down by an unanswerable one"
    assert metrics["faiss/hybrid/abstention-accuracy"] == 1.0
