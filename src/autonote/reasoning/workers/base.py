"""Base class for all reasoning workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from autonote.reasoning.dispatcher import LLMDispatcher


class BaseWorker(ABC):
    """Abstract base class for LLM reasoning workers.

    Each worker knows how to:
    1. Format a prompt for a specific task
    2. Call the dispatcher with the correct task name
    3. Parse the LLM response into structured data
    """

    def __init__(self, dispatcher: LLMDispatcher) -> None:
        self.dispatcher = dispatcher

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Run the worker's reasoning task.

        Args:
            **kwargs: Task-specific inputs (transcript, summary, etc.)

        Returns:
            Parsed result (type depends on the worker).
        """
        ...
