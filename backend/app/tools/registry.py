"""Tool auto-discovery.

Built-in tools live in submodules of `app.tools`; user/community plugins are
Python packages dropped into `app/plugins/`. Both are discovered the same
way: import the module, register every concrete `Tool` subclass found.
There is no separate plugin API.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from types import ModuleType

from app.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning("Tool %s already registered; keeping the first one", tool.name)
            return
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s (%s)", tool.name, tool.risk_level.value)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def register_module_tools(self, module: ModuleType) -> int:
        """Register every concrete Tool subclass defined in `module`."""
        count = 0
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, Tool)
                and obj is not Tool
                and not inspect.isabstract(obj)
                and obj.__module__ == module.__name__
            ):
                self.register(obj())
                count += 1
        return count

    def discover(self) -> None:
        """Import and register built-in tools and plugins."""
        import app.tools as tools_pkg

        self._discover_package(tools_pkg, skip={"base", "registry"})
        try:
            import app.plugins as plugins_pkg
        except ImportError:
            return
        self._discover_package(plugins_pkg, skip=set())

    def _discover_package(self, package: ModuleType, skip: set[str]) -> None:
        for module_info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
            short_name = module_info.name.rsplit(".", 1)[-1]
            if short_name.startswith("_") or short_name in skip:
                continue
            try:
                module = importlib.import_module(module_info.name)
            except ImportError as exc:
                # Tools with missing optional deps (e.g. browser without
                # playwright) are skipped, not fatal.
                logger.info("Skipping %s (%s)", module_info.name, exc)
                continue
            self.register_module_tools(module)
