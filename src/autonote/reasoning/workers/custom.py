"""CustomPromptWorker — runs user's freeform prompt against meeting context."""

from __future__ import annotations

import logging
from typing import Any

from autonote.reasoning.workers.base import BaseWorker
from autonote.realtime.models import CustomPromptResult

logger = logging.getLogger(__name__)


class CustomPromptWorker(BaseWorker):
    """Runs a user-supplied freeform prompt against the full meeting context."""

    async def execute(self, **kwargs: Any) -> CustomPromptResult:
        """Run the custom prompt.

        Args:
            full_context: Full meeting context string.
            user_prompt: The user's freeform question or instruction.
            timestamp: Timestamp to attach to the result (default 0.0).

        Returns:
            CustomPromptResult with the LLM's response.
        """
        full_context: str = kwargs.get("full_context", "")
        user_prompt: str = kwargs.get("user_prompt", "")
        timestamp: float = float(kwargs.get("timestamp", 0.0))

        result = await self.dispatcher.run(
            "custom",
            full_context=full_context or "(No meeting context yet.)",
            user_prompt=user_prompt,
        )

        return CustomPromptResult(
            prompt=user_prompt,
            result=result.strip(),
            timestamp=timestamp,
        )
