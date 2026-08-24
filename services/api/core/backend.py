from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..api.schemas import (
        Agent,
        AgentRun,
        AgentSpec,
        ChatCompletionChunk,
        ChatCompletionRequest,
        ChatCompletionResponse,
        EmbeddingResponse,
        Model,
        RerankResponse,
        Tool,
        ToolInvokeResponse,
    )


class Backend(ABC):
    """Interface implemented by inference/agent backends.

    A production deployment supplies its own backend (e.g. one that talks to a
    model runtime and an agent orchestrator); the default mock backend lets the
    whole API surface be exercised without any model weights.
    """

    @abstractmethod
    def list_models(self) -> list[Model]: ...

    @abstractmethod
    def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...

    @abstractmethod
    def chat_stream(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]: ...

    @abstractmethod
    def embed(self, model: str, inputs: list[str]) -> EmbeddingResponse: ...

    @abstractmethod
    def rerank(
        self, model: str, query: str, documents: list[str], top_n: int | None
    ) -> RerankResponse: ...

    @abstractmethod
    def list_tools(self) -> list[Tool]: ...

    @abstractmethod
    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> ToolInvokeResponse: ...

    @abstractmethod
    def create_agent(self, spec: AgentSpec) -> Agent: ...

    @abstractmethod
    def get_agent(self, agent_id: str) -> Agent: ...

    @abstractmethod
    def list_agents(self) -> list[Agent]: ...

    @abstractmethod
    def delete_agent(self, agent_id: str) -> None: ...

    @abstractmethod
    def run_agent(self, agent_id: str, message: str) -> AgentRun: ...

    @abstractmethod
    def get_run(self, run_id: str) -> AgentRun: ...

    @abstractmethod
    def run_agent_stream(self, agent_id: str, message: str) -> AsyncIterator[dict[str, Any]]: ...


class BackendFactory:
    """Builds the configured backend. Swap via KOTHAGPT_BACKEND env var."""

    def __init__(self) -> None:
        self._registered: dict[str, type[Backend]] = {}
        self._instances: dict[str, Backend] = {}

    def register(self, name: str, backend: type[Backend]) -> None:
        self._registered[name] = backend

    def create(self, name: str | None = None) -> Backend:
        import os

        chosen = name or os.getenv("KOTHAGPT_BACKEND", "mock")
        if chosen not in self._registered:
            raise ValueError(f"Unknown backend {chosen!r}; registered: {sorted(self._registered)}")
        # Backends are shared singletons per process so in-memory state
        # (e.g. agents, tool results) persists across requests.
        if chosen not in self._instances:
            self._instances[chosen] = self._registered[chosen]()
        return self._instances[chosen]
