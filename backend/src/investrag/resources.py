from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum

import httpx


class ResourceProfile(StrEnum):
    """The four operating modes from design §19. Only one is ever active."""

    INTERACTIVE = "interactive"
    PARSER_HEAVY = "parser-heavy"
    DATABASE_BENCHMARK = "database-benchmark"
    MILVUS = "milvus"
    WEAVIATE = "weaviate"


class ResourceComponent(StrEnum):
    """Heavy, VRAM- or resource-contending pieces that a profile can forbid.

    Light, always-resident components (FAISS, the catalog, dense embedding at rest) are
    never listed here — a profile only needs to name what it explicitly forbids.
    """

    GENERATION = "generation"
    MINERU = "mineru"
    MILVUS_STORE = "milvus-store"
    WEAVIATE_STORE = "weaviate-store"
    DATABASE_BENCHMARK_STORE = "database-benchmark-store"


class ResourceConflict(RuntimeError):
    """Raised when a component is requested under a profile that forbids it (design §17)."""

    def __init__(self, component: ResourceComponent, profile: ResourceProfile) -> None:
        self.component = component
        self.profile = profile
        super().__init__(
            f"{component.value} is not available under the '{profile.value}' resource profile"
        )


# What each profile forbids. 8GB VRAM cannot hold generation + MinerU + a heavy service-backed
# store at once (see the completion plan's resource table), so activating one of the heavy
# profiles forbids the others' components rather than merely deprioritising them.
_FORBIDDEN: dict[ResourceProfile, frozenset[ResourceComponent]] = {
    ResourceProfile.INTERACTIVE: frozenset(
        {
            ResourceComponent.MINERU,
            ResourceComponent.MILVUS_STORE,
            ResourceComponent.WEAVIATE_STORE,
            ResourceComponent.DATABASE_BENCHMARK_STORE,
        }
    ),
    ResourceProfile.PARSER_HEAVY: frozenset(
        {
            ResourceComponent.GENERATION,
            ResourceComponent.MILVUS_STORE,
            ResourceComponent.WEAVIATE_STORE,
            ResourceComponent.DATABASE_BENCHMARK_STORE,
        }
    ),
    ResourceProfile.DATABASE_BENCHMARK: frozenset(
        {
            ResourceComponent.MINERU,
            ResourceComponent.MILVUS_STORE,
            ResourceComponent.WEAVIATE_STORE,
        }
    ),
    ResourceProfile.MILVUS: frozenset(
        {
            ResourceComponent.MINERU,
            ResourceComponent.WEAVIATE_STORE,
            ResourceComponent.DATABASE_BENCHMARK_STORE,
        }
    ),
    ResourceProfile.WEAVIATE: frozenset(
        {
            ResourceComponent.MINERU,
            ResourceComponent.MILVUS_STORE,
            ResourceComponent.DATABASE_BENCHMARK_STORE,
        }
    ),
}


@dataclass(frozen=True)
class ResourceSample:
    vram_used_mb: float | None
    vram_total_mb: float | None
    sampled: bool


@dataclass(frozen=True)
class ProfileActivation:
    profile: ResourceProfile
    previous: ResourceProfile
    ollama_unloaded: bool


class ResourceManager:
    """Enforces design §19 so heavy components never contend for the same 8GB card.

    Activating a non-interactive profile unloads the Ollama generation model first
    (`keep_alive: 0`) since it is the single largest resident consumer; returning to
    `interactive` lets it reload lazily on the next query rather than force-loading it.
    """

    def __init__(self, ollama_base_url: str, ollama_model: str) -> None:
        self._profile = ResourceProfile.INTERACTIVE
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._ollama_model = ollama_model
        self._history: list[ProfileActivation] = []

    def current(self) -> ResourceProfile:
        return self._profile

    def assert_allows(self, component: ResourceComponent) -> None:
        if component in _FORBIDDEN.get(self._profile, frozenset()):
            raise ResourceConflict(component, self._profile)

    def allowed_components(self) -> list[ResourceComponent]:
        forbidden = _FORBIDDEN.get(self._profile, frozenset())
        return [c for c in ResourceComponent if c not in forbidden]

    def activate(self, profile: ResourceProfile) -> ProfileActivation:
        previous = self._profile
        self._profile = profile
        unloaded = False
        if profile != ResourceProfile.INTERACTIVE:
            unloaded = self._unload_ollama()
        activation = ProfileActivation(profile=profile, previous=previous, ollama_unloaded=unloaded)
        self._history.append(activation)
        return activation

    def manifest_entries(self) -> list[dict[str, str]]:
        return [
            {
                "profile": entry.profile.value,
                "previous": entry.previous.value,
                "ollama_unloaded": str(entry.ollama_unloaded),
            }
            for entry in self._history
        ]

    def _unload_ollama(self) -> bool:
        try:
            httpx.post(
                f"{self._ollama_base_url}/api/generate",
                json={"model": self._ollama_model, "keep_alive": 0},
                timeout=15,
            )
            return True
        except (httpx.HTTPError, OSError):
            # Best-effort: Ollama may already be down or the model already unloaded.
            # A failed unload must not block the profile switch itself.
            return False

    def sample(self) -> ResourceSample:
        return sample_gpu_memory()


def sample_gpu_memory() -> ResourceSample:
    """Best-effort `nvidia-smi` VRAM sample. Never raises — an unsampleable card must not
    fail a query; it just means the manifest records `sampled: false`."""

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return ResourceSample(None, None, False)
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        used_str, total_str = result.stdout.strip().split(",")
        return ResourceSample(float(used_str.strip()), float(total_str.strip()), True)
    except (subprocess.SubprocessError, OSError, ValueError):
        return ResourceSample(None, None, False)
