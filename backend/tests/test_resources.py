"""Design §19 resource profiles: heavy components must not be usable together on 8GB VRAM."""

from __future__ import annotations

import pytest

from investrag.resources import (
    ResourceComponent,
    ResourceConflict,
    ResourceManager,
    ResourceProfile,
    sample_gpu_memory,
)


def test_interactive_is_the_default_profile() -> None:
    manager = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    assert manager.current() is ResourceProfile.INTERACTIVE


def test_mineru_is_forbidden_under_interactive() -> None:
    manager = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    with pytest.raises(ResourceConflict) as excinfo:
        manager.assert_allows(ResourceComponent.MINERU)
    assert excinfo.value.component is ResourceComponent.MINERU
    assert excinfo.value.profile is ResourceProfile.INTERACTIVE


def test_mineru_is_allowed_after_switching_to_parser_heavy() -> None:
    manager = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    manager.activate(ResourceProfile.PARSER_HEAVY)
    manager.assert_allows(ResourceComponent.MINERU)  # must not raise


def test_generation_is_forbidden_under_parser_heavy() -> None:
    """parser-heavy unloads Ollama, so generation must be refused, not silently degraded."""

    manager = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    manager.activate(ResourceProfile.PARSER_HEAVY)
    with pytest.raises(ResourceConflict):
        manager.assert_allows(ResourceComponent.GENERATION)


@pytest.mark.parametrize(
    ("profile", "forbidden_elsewhere"),
    [
        (ResourceProfile.MILVUS, ResourceComponent.WEAVIATE_STORE),
        (ResourceProfile.WEAVIATE, ResourceComponent.MILVUS_STORE),
    ],
)
def test_milvus_and_weaviate_profiles_are_mutually_exclusive(
    profile: ResourceProfile, forbidden_elsewhere: ResourceComponent
) -> None:
    manager = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    manager.activate(profile)
    with pytest.raises(ResourceConflict):
        manager.assert_allows(forbidden_elsewhere)


def test_activation_records_manifest_history() -> None:
    manager = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    manager.activate(ResourceProfile.PARSER_HEAVY)
    manager.activate(ResourceProfile.INTERACTIVE)
    history = manager.manifest_entries()
    assert [entry["profile"] for entry in history] == ["parser-heavy", "interactive"]
    assert history[0]["previous"] == "interactive"


def test_ollama_unload_is_attempted_on_profile_switch_away_from_interactive() -> None:
    """Even against an unreachable Ollama, activation must not raise - unload is best-effort."""

    manager = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    activation = manager.activate(ResourceProfile.PARSER_HEAVY)
    assert activation.ollama_unloaded is False  # port 1 is unreachable
    assert activation.profile is ResourceProfile.PARSER_HEAVY


def test_returning_to_interactive_does_not_reunload() -> None:
    manager = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    manager.activate(ResourceProfile.PARSER_HEAVY)
    activation = manager.activate(ResourceProfile.INTERACTIVE)
    assert activation.ollama_unloaded is False


def test_gpu_sample_never_raises_when_nvidia_smi_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("investrag.resources.shutil.which", lambda _name: None)
    sample = sample_gpu_memory()
    assert sample.sampled is False
    assert sample.vram_used_mb is None


def test_resource_conflict_becomes_http_409(tmp_path) -> None:
    """The exception handler in main.py must convert ResourceConflict to a typed 409,
    not a 500 crash - proven here since no real endpoint calls assert_allows yet
    (MinerU/Milvus/Weaviate land in later phases)."""

    from fastapi.testclient import TestClient

    from investrag.config import Settings
    from investrag.main import create_app

    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))

    @app.get("/__test_resource_conflict")
    def _trigger() -> None:
        raise ResourceConflict(ResourceComponent.MINERU, ResourceProfile.INTERACTIVE)

    client = TestClient(app)
    response = client.get("/__test_resource_conflict")
    assert response.status_code == 409
    body = response.json()
    assert body["component"] == "mineru"
    assert body["active_profile"] == "interactive"
