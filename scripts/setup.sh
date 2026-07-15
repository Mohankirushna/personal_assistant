#!/usr/bin/env bash
# One-shot development setup for the Jarvis backend.
#
# Usage: scripts/setup.sh [--all] [--with-models]
#   --all           install every optional extra (voice, memory, browser) +
#                   the Playwright browser, and pass --all to install_models.sh
#   --with-models   pull the default Ollama models (see install_models.sh)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_MODELS=false
ALL=false
for arg in "$@"; do
    case "$arg" in
        --with-models) WITH_MODELS=true ;;
        --all) ALL=true; WITH_MODELS=true ;;
    esac
done

missing() { echo "error: '$1' is not installed. $2" >&2; exit 1; }

command -v uv >/dev/null 2>&1 \
    || missing uv "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v ollama >/dev/null 2>&1 \
    || missing ollama "Install it with: brew install ollama  (or download from https://ollama.com)"

if $ALL; then
    echo "==> Installing backend dependencies (all extras)"
    (cd "$REPO_ROOT/backend" && uv sync --extra dev --extra voice --extra memory --extra browser)
    echo "==> Installing Playwright Chromium"
    (cd "$REPO_ROOT/backend" && uv run playwright install chromium)
else
    echo "==> Installing backend dependencies (core + dev)"
    (cd "$REPO_ROOT/backend" && uv sync --extra dev)
fi

if ! ollama list >/dev/null 2>&1; then
    echo "warning: Ollama daemon is not running. Start it with 'ollama serve' or open the Ollama app." >&2
fi

if $WITH_MODELS; then
    $ALL && "$REPO_ROOT/scripts/install_models.sh" --all || "$REPO_ROOT/scripts/install_models.sh"
fi

cat <<'DONE'

Setup complete. Run the backend with:

    cd backend && uv run jarvis-backend

Then check it:

    curl http://127.0.0.1:8765/health
DONE
