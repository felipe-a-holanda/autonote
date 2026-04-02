"""ActionItemWorker — extracts and updates action items from meeting transcript."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from autonote.reasoning.workers.base import BaseWorker
from autonote.realtime.models import ActionItem

logger = logging.getLogger(__name__)


class ActionItemWorker(BaseWorker):
    """Extracts new action items and updates existing ones from the transcript."""

    async def execute(self, **kwargs: Any) -> list[ActionItem]:
        """Run action item extraction.

        Args:
            full_context: Full meeting context string.
            recent_transcript: Recent transcript text.
            existing_items: List of current ActionItem objects.

        Returns:
            Updated list of ActionItem objects.
        """
        full_context: str = kwargs.get("full_context", "")
        recent_transcript: str = kwargs.get("recent_transcript", "")
        existing_items: list[ActionItem] = kwargs.get("existing_items", [])

        existing_items_str = (
            json.dumps([item.model_dump() for item in existing_items], indent=2)
            if existing_items
            else "[]"
        )

        result = await self.dispatcher.run(
            "action_items",
            full_context=full_context,
            recent_transcript=recent_transcript,
            existing_items=existing_items_str,
        )

        return self._parse_response(result, existing_items)

    def _parse_response(
        self, raw: str, existing_items: list[ActionItem]
    ) -> list[ActionItem]:
        """Parse the LLM JSON response into ActionItem objects."""
        try:
            data = self._extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("ActionItemWorker: failed to parse LLM response as JSON")
            return existing_items

        items_by_id = {item.id: item for item in existing_items}

        now = time.time()
        for new in data.get("new_items", []):
            if not isinstance(new, dict) or "description" not in new:
                continue
            item = ActionItem(
                id=str(uuid.uuid4())[:8],
                description=new["description"],
                assignee=new.get("assignee"),
                source_timestamp=now,
                status="new",
            )
            items_by_id[item.id] = item

        for update in data.get("updated_items", []):
            if not isinstance(update, dict) or "id" not in update:
                continue
            item_id = update["id"]
            if item_id in items_by_id:
                existing = items_by_id[item_id]
                new_status = update.get("status", existing.status)
                if new_status in ("new", "updated", "completed"):
                    items_by_id[item_id] = existing.model_copy(
                        update={"status": new_status}
                    )

        return list(items_by_id.values())

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Extract JSON from LLM response, handling markdown code fences."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines[1:] if line.strip() != "```"]
            text = "\n".join(lines)
        return json.loads(text)
