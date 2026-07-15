"""The `Tool` interface every built-in and plugin tool implements.

A tool is: a name, a description the LLM plans against, a Pydantic argument
schema, a risk assessment, and an async `execute`. The registry
(app.tools.registry) discovers subclasses automatically — plugins use exactly
the same base class.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from app.planner.schemas import RiskLevel, ToolResult


class Tool(abc.ABC):
    """Base class for all tools.

    Subclasses set `name`, `description`, `args_model`, and implement
    `run()`. Risk defaults to the class-level `risk_level`; tools whose risk
    depends on the arguments (e.g. terminal commands) override `assess_risk`.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def assess_risk(self, args: BaseModel) -> RiskLevel:
        return self.risk_level

    @abc.abstractmethod
    async def run(self, args: BaseModel) -> ToolResult: ...

    async def execute(self, raw_args: dict[str, Any]) -> ToolResult:
        """Validate raw args against the schema, then run.

        Validation failures come back as failed ToolResults (fed to the LLM
        so it can correct itself) rather than exceptions.
        """
        try:
            args = self.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult.failure(self.name, f"invalid arguments: {exc.errors()}")
        try:
            return await self.run(args)
        except Exception as exc:  # noqa: BLE001 - tool bugs must not kill the planner
            return ToolResult.failure(self.name, f"{type(exc).__name__}: {exc}")

    def parse_args(self, raw_args: dict[str, Any]) -> BaseModel | None:
        """Best-effort parse for risk assessment; None if invalid."""
        try:
            return self.args_model.model_validate(raw_args)
        except ValidationError:
            return None

    def llm_spec(self) -> dict[str, Any]:
        """Compact description of this tool for the planner prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_model.model_json_schema(),
        }
