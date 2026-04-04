"""Pydantic data models for the real-time meeting copilot pipeline.

These models flow through the entire pipeline:
  Recorder -> Transcriber -> ContextManager -> Reasoning Workers -> TUI

Ported from meeting-copilot/backend/ws/protocol.py, stripped of web transport
concerns. The `type` discriminator field is kept for TUI event routing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

class TranscriptSegment(BaseModel):
    """A single transcribed segment with speaker attribution."""

    type: Literal["transcript_segment"] = "transcript_segment"
    speaker: str  # "Me" (mic) or "Them" (system audio)
    text: str
    timestamp_start: float
    timestamp_end: float
    is_partial: bool = False


class AggregatedTurn(BaseModel):
    """Multiple transcript segments merged into a single speaker turn."""

    type: Literal["aggregated_turn"] = "aggregated_turn"
    speaker: str
    text: str
    timestamp_start: float
    timestamp_end: float
    segment_count: int
    wall_time_start: Optional[datetime] = None
    wall_time_end: Optional[datetime] = None

    @field_validator("segment_count")
    @classmethod
    def segment_count_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("segment_count must be >= 1")
        return v


# ---------------------------------------------------------------------------
# Reasoning outputs
# ---------------------------------------------------------------------------

class SummaryUpdate(BaseModel):
    """Progressive summary produced by the reasoning engine."""

    type: Literal["summary_update"] = "summary_update"
    summary: str
    covered_until: float


class ActionItem(BaseModel):
    """A single action item or decision extracted from the meeting."""

    id: str
    description: str
    assignee: Optional[str] = None
    source_timestamp: float
    status: Literal["new", "updated", "completed"] = "new"


class ActionItemsUpdate(BaseModel):
    """Batch update of action items."""

    type: Literal["action_items_update"] = "action_items_update"
    items: list[ActionItem]


class ContradictionAlert(BaseModel):
    """Alert when a contradiction is detected in the transcript."""

    type: Literal["contradiction_alert"] = "contradiction_alert"
    description: str
    statement_a: str
    statement_a_timestamp: float
    statement_b: str
    statement_b_timestamp: float
    severity: Literal["low", "medium", "high"]


class ReplySuggestion(BaseModel):
    """Suggested replies for the user based on meeting context."""

    type: Literal["reply_suggestion"] = "reply_suggestion"
    suggestions: list[str]
    context: str
    triggered_by: Literal["auto", "manual"]


class CustomPromptResult(BaseModel):
    """Result of a user's freeform prompt against meeting context."""

    type: Literal["custom_prompt_result"] = "custom_prompt_result"
    prompt: str
    result: str
    timestamp: float


# ---------------------------------------------------------------------------
# Union type for TUI event routing
# ---------------------------------------------------------------------------

RealtimeEvent = (
    TranscriptSegment
    | AggregatedTurn
    | SummaryUpdate
    | ActionItemsUpdate
    | ContradictionAlert
    | ReplySuggestion
    | CustomPromptResult
)
