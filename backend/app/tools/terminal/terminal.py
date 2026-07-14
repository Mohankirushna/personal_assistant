"""Terminal tool: execute shell commands, run python/git/npm/yarn/brew/docker. Classifies commands
(e.g. 'rm -rf', 'brew uninstall') as risk_level=DESTRUCTIVE via a pattern-matching policy before
execution.

Phase 1 stub. Implemented in Phase 6 (Tools). Implements the `Tool` interface
from app.tools.base.
"""

from __future__ import annotations
