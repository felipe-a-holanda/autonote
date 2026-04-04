"""CoachWorker — proactive mission-aware coaching suggestions."""

from __future__ import annotations

import json
import logging
from typing import Any

from autonote.reasoning.workers.base import BaseWorker
from autonote.realtime.models import CoachSuggestion

logger = logging.getLogger(__name__)


class CoachWorker(BaseWorker):
    """Suggests relevant arguments and actions based on a pre-loaded mission brief."""

    async def execute(self, **kwargs: Any) -> CoachSuggestion:
        """Run coach suggestion generation.

        Args:
            full_context: Full meeting context string.
            recent_transcript: The most recent speaker turns.
            mission_brief_text: Formatted MissionBrief text for prompt injection.
            timestamp: Timestamp to attach to the result.

        Returns:
            CoachSuggestion with action guidance and confidence level.
        """
        full_context: str = kwargs.get("full_context", "")
        recent_transcript: str = kwargs.get("recent_transcript", "")
        mission_brief_text: str = kwargs.get("mission_brief_text", "")
        timestamp: float = float(kwargs.get("timestamp", 0.0))

        result = await self.dispatcher.run(
            "coach",
            mission_brief=mission_brief_text,
            full_context=full_context or "(No meeting context yet.)",
            recent_transcript=recent_transcript or "(No transcript yet.)",
        )

        return self._parse_response(result, timestamp)

    def _parse_response(self, raw: str, timestamp: float) -> CoachSuggestion:
        """Parse the LLM JSON response into a CoachSuggestion."""
        try:
            data = self._extract_json(raw)
            should_speak = bool(data.get("should_speak", False))
            suggestion = str(data.get("suggestion", "Stay quiet and listen.")).strip()
            argument_used = data.get("argument_used") or None
            if argument_used is not None:
                argument_used = str(argument_used).strip() or None
            reasoning = str(data.get("reasoning", "")).strip()
            confidence = data.get("confidence", "low")
            if confidence not in ("high", "medium", "low"):
                confidence = "low"
        except (json.JSONDecodeError, ValueError):
            logger.warning("CoachWorker: failed to parse LLM response as JSON")
            return CoachSuggestion(
                should_speak=False,
                suggestion="Stay quiet and listen.",
                argument_used=None,
                reasoning="(Could not parse coach response.)",
                confidence="low",
                timestamp=timestamp,
            )

        return CoachSuggestion(
            should_speak=should_speak,
            suggestion=suggestion,
            argument_used=argument_used,
            reasoning=reasoning,
            confidence=confidence,
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
