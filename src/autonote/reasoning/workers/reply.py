"""ReplyWorker — generates reply suggestions based on meeting context."""

from __future__ import annotations

import json
import logging
from typing import Any

from autonote.reasoning.workers.base import BaseWorker
from autonote.realtime.models import ReplySuggestion

logger = logging.getLogger(__name__)


class ReplyWorker(BaseWorker):
    """Generates 2-3 reply suggestions the user could say in the current meeting."""

    async def execute(self, **kwargs: Any) -> ReplySuggestion:
        """Run reply suggestion generation.

        Args:
            full_context: Full meeting context string.
            context_hint: Optional hint about what to respond to.

        Returns:
            ReplySuggestion with a list of suggestions and a context note.
        """
        full_context: str = kwargs.get("full_context", "")
        context_hint: str = kwargs.get("context_hint", "No specific context provided.")

        result = await self.dispatcher.run(
            "reply",
            full_context=full_context or "(No meeting context yet.)",
            context_hint=context_hint,
        )

        return self._parse_response(result, context_hint)

    def _parse_response(self, raw: str, context_hint: str) -> ReplySuggestion:
        """Parse the LLM JSON response into a ReplySuggestion."""
        try:
            data = self._extract_json(raw)
            suggestions = data.get("suggestions", [])
            if not isinstance(suggestions, list):
                suggestions = []
            suggestions = [s for s in suggestions if isinstance(s, str) and s.strip()]
            context = data.get("context", "")
            if not isinstance(context, str):
                context = ""
        except (json.JSONDecodeError, ValueError):
            logger.warning("ReplyWorker: failed to parse LLM response as JSON")
            stripped = raw.strip()
            suggestions = [stripped] if stripped else []
            context = context_hint

        return ReplySuggestion(
            suggestions=suggestions,
            context=context,
            triggered_by="manual",
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
