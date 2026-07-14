#!/usr/bin/env bash
# Pull the local models Jarvis uses. Only the default LLM is required;
# everything else is opt-in per phase.
#
# Usage: scripts/install_models.sh [--power] [--vision] [--embeddings] [--all]
#   (no flags)     default LLM only            (~2.0 GB)
#   --power        + 7B "power mode" LLM       (~4.7 GB)
#   --vision       + Qwen2.5-VL 3B, Phase 8    (~3.2 GB)
#   --embeddings   + nomic-embed-text, Phase 7 (~0.3 GB)
#   --all          everything above
#
# Speech models (whisper.cpp, Piper voices, openWakeWord) are added here in
# Phase 4 (Voice).
set -euo pipefail

DEFAULT_LLM="qwen2.5:3b-instruct-q4_K_M"
POWER_LLM="qwen2.5:7b-instruct-q4_K_M"
VISION_MODEL="qwen2.5vl:3b"
EMBED_MODEL="nomic-embed-text"

POWER=false; VISION=false; EMBED=false
for arg in "$@"; do
    case "$arg" in
        --power) POWER=true ;;
        --vision) VISION=true ;;
        --embeddings) EMBED=true ;;
        --all) POWER=true; VISION=true; EMBED=true ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

command -v ollama >/dev/null 2>&1 || { echo "error: ollama is not installed" >&2; exit 1; }
ollama list >/dev/null 2>&1 || { echo "error: Ollama daemon is not running (try 'ollama serve')" >&2; exit 1; }

pull() { echo "==> ollama pull $1"; ollama pull "$1"; }

pull "$DEFAULT_LLM"
$POWER  && pull "$POWER_LLM"
$VISION && pull "$VISION_MODEL"
$EMBED  && pull "$EMBED_MODEL"

echo "Done. Installed models:"
ollama list
