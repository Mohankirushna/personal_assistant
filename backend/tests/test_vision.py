"""Vision: service model-swap behavior, tool wiring, and app registration."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.model_manager import HeavyModelKind, ModelManager
from app.main import create_app
from app.vision.qwen_vl import VisionService
from tests.conftest import FakeOllamaClient


def write_png(path: Path, rgb: tuple[int, int, int] = (255, 0, 0), size: int = 64) -> None:
    """Hand-rolled solid-color PNG (no imaging dependency needed)."""
    row = b"\x00" + bytes(rgb) * size
    raw = row * size

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class VisionFake(FakeOllamaClient):
    def __init__(self) -> None:
        super().__init__()
        self.image_prompts: list[tuple[str, str]] = []

    async def chat_with_image(
        self,
        model: str,
        prompt: str,
        image_path: str,
        keep_alive: str | int,
        options: dict | None = None,
    ) -> str:
        self.calls.append(("chat_with_image", model))
        self.image_prompts.append((prompt, image_path))
        return "A code editor with an error dialog."


async def test_describe_swaps_models(settings: Settings, tmp_path: Path) -> None:
    fake = VisionFake()
    manager = ModelManager(fake, settings)
    await manager.ensure_llm()  # simulate normal chat state
    service = VisionService(fake, manager, settings)

    image = tmp_path / "screen.png"
    write_png(image)
    description = await service.describe_image(image)

    assert description == "A code editor with an error dialog."
    assert manager.current_kind is HeavyModelKind.VISION
    # LLM was unloaded before the vision model loaded.
    assert fake.calls.index(("unload", settings.llm_model)) < fake.calls.index(
        ("load", settings.vision_model)
    )
    # Vision inference used the vision model.
    assert ("chat_with_image", settings.vision_model) in fake.calls

    # Next chat turn swaps the LLM back in.
    await manager.ensure_llm()
    assert manager.current_kind is HeavyModelKind.LLM


async def test_describe_missing_image(settings: Settings, tmp_path: Path) -> None:
    service = VisionService(
        VisionFake(), ModelManager(VisionFake(), settings), settings
    )
    with pytest.raises(FileNotFoundError):
        await service.describe_image(tmp_path / "nope.png")


def test_app_registers_look_at_screen(settings: Settings) -> None:
    """The default app wires the vision tool with its service injected."""
    app = create_app(settings=settings, ollama_client=VisionFake(), enable_memory=False)
    with TestClient(app) as client:
        names = {tool["name"] for tool in client.get("/tools").json()}
    assert "look_at_screen" in names


@pytest.mark.integration
async def test_real_vision_inference(tmp_path: Path) -> None:
    """Real Qwen2.5-VL round trip, including the RAM swap, on a generated image."""
    from app.core.ollama_client import OllamaClient

    settings = Settings(_env_file=None)
    client = OllamaClient(host=settings.ollama_host, timeout_seconds=180)
    manager = ModelManager(client, settings)
    try:
        image = tmp_path / "red.png"
        write_png(image, rgb=(255, 0, 0))
        service = VisionService(client, manager, settings)
        description = await service.describe_image(
            image, question="What single solid color fills this image? One word."
        )
        assert description.strip(), "vision model returned nothing"
        assert "red" in description.lower(), f"expected 'red', got: {description!r}"
    finally:
        await manager.release_all()
        await client.aclose()
