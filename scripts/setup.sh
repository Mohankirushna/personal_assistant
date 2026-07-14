#!/usr/bin/env bash
# One-shot development setup for the Jarvis backend.
#
# Usage: scripts/setup.sh [--with-models]
#   --with-models   also pull the default Ollama models (see install_models.sh)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_MODELS=false
[[ "${1:-}" == "--with-models" ]] && WITH_MODELS=true

missing() { echo "error: '$1' is not installed. $2" >&2; exit 1; }

command -v uv >/dev/null 2>&1 \
    || missing uv "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v ollama >/dev/null 2>&1 \
    || missing ollama "Install it with: brew install ollama  (or download from https://ollama.com)"

echo "==> Installing backend dependencies (uv sync)"
(cd "$REPO_ROOT/backend" && uv sync --extra dev)

if ! ollama list >/dev/null 2>&1; then
    echo "warning: Ollama daemon is not running. Start it with 'ollama serve' or open the Ollama app." >&2
fi

if $WITH_MODELS; then
    "$REPO_ROOT/scripts/install_models.sh"
fi

cat <<'DONE'

Setup complete. Run the backend with:

    cd backend && uv run jarvis-backend

Then check it:

    curl http://127.0.0.1:8765/health
DONE
