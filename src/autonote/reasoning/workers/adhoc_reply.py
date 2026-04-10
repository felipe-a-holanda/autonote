"""AdhocReplyWorker — generates a single strategic reply using the smartest model."""

from __future__ import annotations

import json
import logging
from typing import Any

from autonote.reasoning.workers.base import BaseWorker
from autonote.realtime.models import AdhocReplySuggestion

logger = logging.getLogger(__name__)


class AdhocReplyWorker(BaseWorker):
    """Generates a single detailed reply suggestion using the smartest available model."""

    async def execute(self, **kwargs: Any) -> AdhocReplySuggestion:
        """Run ad-hoc reply generation.

        Args:
            full_context: Full meeting context string (unwindowed).
            mission_section: Formatted mission brief or empty string.
            timestamp: Current transcript timestamp.

        Returns:
            AdhocReplySuggestion with reply text and reasoning.
        """
        full_context: str = kwargs.get("full_context", "")
        mission_section: str = kwargs.get("mission_section", "")
        timestamp: float = kwargs.get("timestamp", 0.0)
        model_override: str | None = kwargs.get("model_override")

        result = await self.dispatcher.run(
            "adhoc_reply",
            model_override=model_override,
            full_context=full_context or "(No meeting context yet.)",
            mission_section=mission_section,
        )

        return self._parse_response(result, timestamp)

    def _parse_response(self, raw: str, timestamp: float) -> AdhocReplySuggestion:
        """Parse the LLM JSON response into an AdhocReplySuggestion."""
        try:
            data = self._extract_json(raw)
            reply = data.get("reply", "")
            reasoning = data.get("reasoning", "")
            if not isinstance(reply, str) or not reply.strip():
                reply = raw.strip()
                reasoning = ""
        except (json.JSONDecodeError, ValueError):
            logger.warning("AdhocReplyWorker: failed to parse LLM response as JSON")
            reply = raw.strip()
            reasoning = ""

        return AdhocReplySuggestion(
            reply=reply,
            reasoning=reasoning,
            timestamp=timestamp,
        )

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Extract JSON from LLM response, handling markdown code fences."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines[1:] if line.strip() != "```"]
            text = "\n".join(lines)
        return json.loads(text)
