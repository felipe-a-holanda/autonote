"""SummaryWorker — progressive meeting summarization."""

from __future__ import annotations

import logging
from typing import Any

from autonote.reasoning.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class SummaryWorker(BaseWorker):
    """Takes current summary + new transcript segments and returns an updated summary."""

    async def execute(self, **kwargs: Any) -> str:
        """Run progressive summarization.

        Args:
            current_summary: The existing summary text (empty string if first run).
            new_segments: Formatted transcript of new segments since last summary.

        Returns:
            Updated summary string.
        """
        current_summary: str = kwargs.get("current_summary", "")
        new_segments: str = kwargs.get("new_segments", "")

        if not new_segments.strip():
            logger.debug("SummaryWorker: no new segments, returning current summary")
            return current_summary

        result = await self.dispatcher.run(
            "summary",
            current_summary=current_summary or "(No summary yet — this is the first update.)",
            new_segments=new_segments,
        )
        return result.strip()
