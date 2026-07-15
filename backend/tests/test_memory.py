"""Memory: SQLite store, Chroma vector store (fake embedder), service facade,
API endpoints, and planner context injection."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.memory.service import MemoryService
from app.memory.store import MemoryStore
from app.memory.vector_store import VectorStore
from app.planner.schemas import PlanExecution, PlanStep, RiskLevel, ToolResult
from tests.conftest import FakeOllamaClient
from tests.test_planner import respond, tool_call  # ChatTurn builders


class FakeEmbedder:
    """Keyword-direction vectors (matches FakeOllamaClient.embed)."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "alpha" in lowered else 0.0,
                    1.0 if "beta" in lowered else 0.0,
                    1.0 if "gamma" in lowered else 0.0,
                    0.1,
                ]
            )
        return vectors


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


@pytest.fixture
def vectors(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", FakeEmbedder())


@pytest.fixture
def service(store: MemoryStore, vectors: VectorStore) -> MemoryService:
    return MemoryService(store, vectors, context_hits=2)


def execution(utterance: str, reply: str) -> PlanExecution:
    return PlanExecution(
        utterance=utterance,
        reply=reply,
        steps=[
            PlanStep(
                tool="clock",
                args={},
                risk=RiskLevel.SAFE,
                result=ToolResult(tool="clock", ok=True, summary="3pm"),
            )
        ],
    )


async def test_store_records_and_lists(store: MemoryStore) -> None:
    await store.record_turn("s1", "hello", "hi", [{"tool": "clock"}])
    await store.record_turn("s1", "second", "reply2", [])
    entries = await store.history(limit=10)
    assert len(entries) == 2
    assert entries[0].utterance == "second"  # newest first
    assert entries[1].steps == [{"tool": "clock"}]


async def test_preferences_roundtrip(store: MemoryStore) -> None:
    assert await store.get_preference("editor") is None
    await store.set_preference("editor", "vscode")
    await store.set_preference("editor", "zed")  # upsert
    assert await store.get_preference("editor") == "zed"


async def test_projects(store: MemoryStore) -> None:
    await store.touch_project("/tmp/a", "a")
    await store.touch_project("/tmp/b", "b")
    await store.touch_project("/tmp/a", "a")  # bump recency
    projects = await store.recent_projects()
    assert projects[0] == ("/tmp/a", "a")


async def test_vector_search_relevance(vectors: VectorStore) -> None:
    await vectors.add_turn("user opened the alpha project")
    await vectors.add_turn("user asked about beta testing")
    hits = await vectors.search("tell me about alpha", k=2)
    assert hits[0].text == "user opened the alpha project"
    assert hits[0].distance < hits[1].distance


async def test_service_records_and_recalls(service: MemoryService) -> None:
    await service.record_turn("s1", execution("open the alpha project", "Opened alpha."))
    context = await service.context_for("what do you know about alpha?")
    assert context is not None
    assert "alpha" in context

    # Unrelated queries recall nothing (distance above threshold).
    assert await service.context_for("something about gamma rays") is None


async def test_empty_vector_store_searches_cleanly(vectors: VectorStore) -> None:
    assert await vectors.search("anything") == []


def test_memory_api_and_planner_context(settings: Settings, tmp_path: Path) -> None:
    """Full wiring: a planner turn is recorded, visible in /memory endpoints,
    and recalled into the next turn's system prompt."""
    from app.tools.registry import ToolRegistry
    from tests.test_planner import EchoTool

    fake = FakeOllamaClient()
    fake.queued_turns = [
        tool_call("echo", text="alpha"),
        respond("Echoed alpha for you."),
        respond("You asked me to echo alpha earlier."),
    ]
    registry = ToolRegistry()
    registry.register(EchoTool())
    memory = MemoryService(
        MemoryStore(tmp_path / "m.db"),
        VectorStore(tmp_path / "chroma", FakeEmbedder()),
    )
    app = create_app(
        settings=settings, ollama_client=fake, registry=registry, memory=memory
    )
    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "please echo alpha"})
        assert first.status_code == 200

        history = client.get("/memory/history").json()
        assert len(history) == 1
        assert history[0]["utterance"] == "please echo alpha"
        assert history[0]["steps"][0]["tool"] == "echo"

        hits = client.get("/memory/search", params={"q": "alpha"}).json()
        assert len(hits) == 1

        second = client.post("/chat", json={"message": "what did I ask about alpha?"})
        assert second.status_code == 200
    # The second turn's system prompt carried the recalled context.
    system = fake.chat_messages[-1][0]
    assert system["role"] == "system"
    assert "Possibly relevant past interactions" in system["content"]
    assert "echo alpha" in system["content"].lower()


@pytest.mark.integration
async def test_semantic_recall_with_real_embeddings(tmp_path: Path) -> None:
    """nomic-embed-text via Ollama: recall must work on meaning, not keywords."""
    from app.core.ollama_client import OllamaClient
    from app.memory.embeddings import OllamaEmbedder

    settings = Settings(_env_file=None)
    client = OllamaClient(host=settings.ollama_host, timeout_seconds=60)
    try:
        vectors = VectorStore(tmp_path / "chroma", OllamaEmbedder(client))
        await vectors.add_turn(
            "User asked: open the dashboard project in VS Code\n"
            "Outcome (vscode_open): Opened ~/projects/dashboard in VS Code"
        )
        await vectors.add_turn(
            "User asked: what's the weather like\n"
            "Outcome (no tools): I can't check the weather yet."
        )
        # No shared keywords with the first turn besides meaning.
        hits = await vectors.search("which coding workspace did I use recently?", k=2)
        assert hits[0].text.startswith("User asked: open the dashboard project")
    finally:
        await client.aclose()
