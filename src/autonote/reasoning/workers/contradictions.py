"""ContradictionWorker — detects contradictions and inconsistencies in meeting transcript."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from autonote.reasoning.workers.base import BaseWorker
from autonote.realtime.models import ContradictionAlert

logger = logging.getLogger(__name__)


class ContradictionWorker(BaseWorker):
    """Analyzes recent transcript against summary to detect contradictions."""

    async def execute(self, **kwargs: Any) -> list[ContradictionAlert]:
        """Run contradiction detection.

        Args:
            current_summary: The meeting summary so far.
            recent_transcript: Recent transcript text to check.

        Returns:
            List of ContradictionAlert objects (may be empty).
        """
        current_summary: str = kwargs.get("current_summary", "")
        recent_transcript: str = kwargs.get("recent_transcript", "")

        if not recent_transcript.strip():
            logger.debug("ContradictionWorker: no recent transcript, skipping")
            return []

        result = await self.dispatcher.run(
            "contradictions",
            current_summary=current_summary or "(No summary yet.)",
            recent_transcript=recent_transcript,
        )

        return self._parse_response(result)

    def _parse_response(self, raw: str) -> list[ContradictionAlert]:
        """Parse the LLM JSON response into ContradictionAlert objects."""
        try:
            data = self._extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("ContradictionWorker: failed to parse LLM response as JSON")
            return []

        contradictions: list[ContradictionAlert] = []
        now = time.time()

        for item in data.get("contradictions", []):
            if not isinstance(item, dict) or "description" not in item:
                continue
            severity = item.get("severity", "low")
            if severity not in ("low", "medium", "high"):
                severity = "low"
            contradictions.append(
                ContradictionAlert(
                    description=item["description"],
                    statement_a=item.get("statement_a", ""),
                    statement_a_timestamp=now,
                    statement_b=item.get("statement_b", ""),
                    statement_b_timestamp=now,
                    severity=severity,
                )
            )

        return contradictions

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Extract JSON from LLM response, handling markdown code fences."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines[1:] if line.strip() != "```"]
            text = "\n".join(lines)
        return json.loads(text)
