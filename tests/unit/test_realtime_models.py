"""Tests for realtime event models."""

import pytest
from pydantic import ValidationError
from autonote.realtime.models import (
    TranscriptSegment,
    SummaryUpdate,
    ActionItem,
    ActionItemsUpdate,
    ContradictionAlert,
    ReplySuggestion,
    CustomPromptResult,
)


class TestTranscriptSegment:
    """Test TranscriptSegment model."""

    def test_valid_transcript_segment(self):
        """Test creating a valid transcript segment."""
        segment = TranscriptSegment(
            speaker="Me",
            text="Hello, how are you?",
            timestamp_start=0.0,
            timestamp_end=2.5,
        )
        assert segment.type == "transcript_segment"
        assert segment.speaker == "Me"
        assert segment.text == "Hello, how are you?"
        assert segment.timestamp_start == 0.0
        assert segment.timestamp_end == 2.5
        assert segment.is_partial is False

    def test_partial_transcript_segment(self):
        """Test creating a partial transcript segment."""
        segment = TranscriptSegment(
            speaker="Them",
            text="Good morning",
            timestamp_start=2.5,
            timestamp_end=4.0,
            is_partial=True,
        )
        assert segment.is_partial is True

    def test_transcript_segment_type_discrimination(self):
        """Test that type field is always set correctly."""
        segment = TranscriptSegment(
            speaker="Me",
            text="Test",
            timestamp_start=0.0,
            timestamp_end=1.0,
        )
        assert segment.type == "transcript_segment"

    def test_transcript_segment_missing_required_fields(self):
        """Test that creating segment without required fields fails."""
        with pytest.raises(ValidationError):
            TranscriptSegment(speaker="Me", text="Hello")  # Missing timestamps


class TestSummaryUpdate:
    """Test SummaryUpdate model."""

    def test_valid_summary_update(self):
        """Test creating a valid summary update."""
        update = SummaryUpdate(
            summary="Meeting discussed Q2 plans",
            covered_until=120.5,
        )
        assert update.type == "summary_update"
        assert update.summary == "Meeting discussed Q2 plans"
        assert update.covered_until == 120.5

    def test_empty_summary(self):
        """Test that empty summaries are allowed."""
        update = SummaryUpdate(summary="", covered_until=0.0)
        assert update.summary == ""

    def test_summary_update_missing_fields(self):
        """Test that creating update without required fields fails."""
        with pytest.raises(ValidationError):
            SummaryUpdate(summary="Test")  # Missing covered_until


class TestActionItem:
    """Test ActionItem model."""

    def test_valid_action_item(self):
        """Test creating a valid action item."""
        item = ActionItem(
            id="task-001",
            description="Implement authentication",
            assignee="Alice",
            source_timestamp=45.0,
            status="new",
        )
        assert item.id == "task-001"
        assert item.description == "Implement authentication"
        assert item.assignee == "Alice"
        assert item.source_timestamp == 45.0
        assert item.status == "new"

    def test_action_item_without_assignee(self):
        """Test creating action item without assignee."""
        item = ActionItem(
            id="task-002",
            description="Review PR",
            source_timestamp=50.0,
        )
        assert item.assignee is None
        assert item.status == "new"

    def test_action_item_updated_status(self):
        """Test action item with updated status."""
        item = ActionItem(
            id="task-003",
            description="Fix bug",
            source_timestamp=60.0,
            status="updated",
        )
        assert item.status == "updated"

    def test_action_item_completed_status(self):
        """Test action item with completed status."""
        item = ActionItem(
            id="task-004",
            description="Deploy to prod",
            source_timestamp=70.0,
            status="completed",
        )
        assert item.status == "completed"

    def test_action_item_invalid_status(self):
        """Test that invalid status values are rejected."""
        with pytest.raises(ValueError):
            ActionItem(
                id="task-005",
                description="Test",
                source_timestamp=0.0,
                status="invalid",  # type: ignore
            )


class TestActionItemsUpdate:
    """Test ActionItemsUpdate model."""

    def test_valid_action_items_update(self):
        """Test creating a valid action items update."""
        items = [
            ActionItem(
                id="task-001",
                description="Task 1",
                source_timestamp=10.0,
            ),
            ActionItem(
                id="task-002",
                description="Task 2",
                assignee="Bob",
                source_timestamp=20.0,
            ),
        ]
        update = ActionItemsUpdate(items=items)
        assert update.type == "action_items_update"
        assert len(update.items) == 2
        assert update.items[0].description == "Task 1"

    def test_empty_action_items_update(self):
        """Test action items update with empty list."""
        update = ActionItemsUpdate(items=[])
        assert len(update.items) == 0

    def test_action_items_update_single_item(self):
        """Test action items update with single item."""
        item = ActionItem(
            id="task-001",
            description="Single task",
            source_timestamp=5.0,
        )
        update = ActionItemsUpdate(items=[item])
        assert len(update.items) == 1


class TestContradictionAlert:
    """Test ContradictionAlert model."""

    def test_valid_contradiction_alert(self):
        """Test creating a valid contradiction alert."""
        alert = ContradictionAlert(
            description="Conflicting statements about timeline",
            statement_a="Project deadline is May 1st",
            statement_a_timestamp=30.0,
            statement_b="Project deadline is May 15th",
            statement_b_timestamp=120.0,
            severity="high",
        )
        assert alert.type == "contradiction_alert"
        assert alert.description == "Conflicting statements about timeline"
        assert alert.severity == "high"

    def test_contradiction_alert_low_severity(self):
        """Test contradiction alert with low severity."""
        alert = ContradictionAlert(
            description="Minor difference",
            statement_a="Statement A",
            statement_a_timestamp=10.0,
            statement_b="Statement B",
            statement_b_timestamp=20.0,
            severity="low",
        )
        assert alert.severity == "low"

    def test_contradiction_alert_medium_severity(self):
        """Test contradiction alert with medium severity."""
        alert = ContradictionAlert(
            description="Moderate difference",
            statement_a="Statement A",
            statement_a_timestamp=10.0,
            statement_b="Statement B",
            statement_b_timestamp=20.0,
            severity="medium",
        )
        assert alert.severity == "medium"

    def test_contradiction_alert_invalid_severity(self):
        """Test that invalid severity is rejected."""
        with pytest.raises(ValueError):
            ContradictionAlert(
                description="Test",
                statement_a="A",
                statement_a_timestamp=0.0,
                statement_b="B",
                statement_b_timestamp=1.0,
                severity="critical",  # type: ignore
            )


class TestReplySuggestion:
    """Test ReplySuggestion model."""

    def test_valid_reply_suggestion_auto(self):
        """Test creating a valid auto-triggered reply suggestion."""
        suggestion = ReplySuggestion(
            suggestions=[
                "That sounds good, let's proceed",
                "I need more time to review",
            ],
            context="Discussion about implementation approach",
            triggered_by="auto",
        )
        assert suggestion.type == "reply_suggestion"
        assert len(suggestion.suggestions) == 2
        assert suggestion.triggered_by == "auto"

    def test_valid_reply_suggestion_manual(self):
        """Test creating a valid manually-triggered reply suggestion."""
        suggestion = ReplySuggestion(
            suggestions=["Can you clarify that point?"],
            context="Unclear requirement",
            triggered_by="manual",
        )
        assert suggestion.triggered_by == "manual"

    def test_reply_suggestion_empty_suggestions(self):
        """Test reply suggestion with no suggestions."""
        suggestion = ReplySuggestion(
            suggestions=[],
            context="No good suggestions",
            triggered_by="auto",
        )
        assert len(suggestion.suggestions) == 0

    def test_reply_suggestion_invalid_trigger(self):
        """Test that invalid trigger type is rejected."""
        with pytest.raises(ValueError):
            ReplySuggestion(
                suggestions=["Test"],
                context="Test",
                triggered_by="invalid",  # type: ignore
            )


class TestCustomPromptResult:
    """Test CustomPromptResult model."""

    def test_valid_custom_prompt_result(self):
        """Test creating a valid custom prompt result."""
        result = CustomPromptResult(
            prompt="What are the main action items?",
            result="The main action items are: 1) Review API design 2) Implement caching",
            timestamp=90.5,
        )
        assert result.type == "custom_prompt_result"
        assert result.prompt == "What are the main action items?"
        assert "Review API design" in result.result

    def test_custom_prompt_result_empty_result(self):
        """Test custom prompt result with empty result."""
        result = CustomPromptResult(
            prompt="Test question?",
            result="",
            timestamp=0.0,
        )
        assert result.result == ""

    def test_custom_prompt_result_missing_fields(self):
        """Test that missing required fields fail."""
        with pytest.raises(ValidationError):
            CustomPromptResult(
                prompt="Test",
                # Missing result and timestamp
            )


class TestRealtimeEventUnion:
    """Test that models serialize/deserialize correctly for event routing."""

    def test_transcript_segment_serialization(self):
        """Test transcript segment can be serialized."""
        segment = TranscriptSegment(
            speaker="Me",
            text="Test message",
            timestamp_start=0.0,
            timestamp_end=1.0,
        )
        data = segment.model_dump()
        assert data["type"] == "transcript_segment"
        assert data["speaker"] == "Me"

    def test_summary_update_serialization(self):
        """Test summary update can be serialized."""
        update = SummaryUpdate(
            summary="Test summary",
            covered_until=50.0,
        )
        data = update.model_dump()
        assert data["type"] == "summary_update"

    def test_action_items_update_serialization(self):
        """Test action items update can be serialized."""
        items = [
            ActionItem(
                id="task-001",
                description="Test",
                source_timestamp=10.0,
            )
        ]
        update = ActionItemsUpdate(items=items)
        data = update.model_dump()
        assert data["type"] == "action_items_update"
        assert len(data["items"]) == 1

    def test_contradiction_alert_serialization(self):
        """Test contradiction alert can be serialized."""
        alert = ContradictionAlert(
            description="Test",
            statement_a="A",
            statement_a_timestamp=0.0,
            statement_b="B",
            statement_b_timestamp=1.0,
            severity="low",
        )
        data = alert.model_dump()
        assert data["type"] == "contradiction_alert"

    def test_reply_suggestion_serialization(self):
        """Test reply suggestion can be serialized."""
        suggestion = ReplySuggestion(
            suggestions=["Test"],
            context="Test context",
            triggered_by="auto",
        )
        data = suggestion.model_dump()
        assert data["type"] == "reply_suggestion"

    def test_custom_prompt_result_serialization(self):
        """Test custom prompt result can be serialized."""
        result = CustomPromptResult(
            prompt="Test?",
            result="Answer",
            timestamp=5.0,
        )
        data = result.model_dump()
        assert data["type"] == "custom_prompt_result"
