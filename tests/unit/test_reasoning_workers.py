"""Tests for reasoning worker classes."""

import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from autonote.reasoning.workers.action_items import ActionItemWorker
from autonote.reasoning.workers.contradictions import ContradictionWorker
from autonote.reasoning.workers.reply import ReplyWorker
from autonote.reasoning.workers.summary import SummaryWorker
from autonote.reasoning.workers.custom import CustomPromptWorker
from autonote.realtime.models import ActionItem, ContradictionAlert, ReplySuggestion, CustomPromptResult


def make_dispatcher(return_value: str = "{}") -> MagicMock:
    dispatcher = MagicMock()
    dispatcher.run = AsyncMock(return_value=return_value)
    return dispatcher


# ---------------------------------------------------------------------------
# ActionItemWorker
# ---------------------------------------------------------------------------

class TestActionItemWorkerExtractJson:

    def test_plain_json(self):
        raw = '{"new_items": [], "updated_items": []}'
        result = ActionItemWorker._extract_json(raw)
        assert result == {"new_items": [], "updated_items": []}

    def test_markdown_fenced_json(self):
        raw = '```json\n{"new_items": [], "updated_items": []}\n```'
        result = ActionItemWorker._extract_json(raw)
        assert result == {"new_items": [], "updated_items": []}

    def test_fenced_without_lang(self):
        raw = '```\n{"new_items": [], "updated_items": []}\n```'
        result = ActionItemWorker._extract_json(raw)
        assert result == {"new_items": [], "updated_items": []}

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            ActionItemWorker._extract_json("not json at all")


class TestActionItemWorkerParseResponse:

    def _make_worker(self):
        return ActionItemWorker(make_dispatcher())

    def test_parses_new_items(self):
        worker = self._make_worker()
        raw = json.dumps({
            "new_items": [{"description": "Write tests", "assignee": "Alice"}],
            "updated_items": [],
        })
        result = worker._parse_response(raw, [])
        assert len(result) == 1
        assert result[0].description == "Write tests"
        assert result[0].assignee == "Alice"
        assert result[0].status == "new"

    def test_parses_updated_items(self):
        worker = self._make_worker()
        existing = [ActionItem(id="abc123", description="Fix bug", source_timestamp=1.0)]
        raw = json.dumps({
            "new_items": [],
            "updated_items": [{"id": "abc123", "status": "completed"}],
        })
        result = worker._parse_response(raw, existing)
        assert len(result) == 1
        assert result[0].status == "completed"

    def test_invalid_json_returns_existing(self):
        worker = self._make_worker()
        existing = [ActionItem(id="x", description="Keep me", source_timestamp=1.0)]
        result = worker._parse_response("not json", existing)
        assert result == existing

    def test_skips_new_item_without_description(self):
        worker = self._make_worker()
        raw = json.dumps({
            "new_items": [{"assignee": "Bob"}],  # no description
            "updated_items": [],
        })
        result = worker._parse_response(raw, [])
        assert result == []

    def test_invalid_status_not_applied(self):
        worker = self._make_worker()
        existing = [ActionItem(id="abc", description="task", source_timestamp=1.0, status="new")]
        raw = json.dumps({
            "new_items": [],
            "updated_items": [{"id": "abc", "status": "invalid_status"}],
        })
        result = worker._parse_response(raw, existing)
        # Status should remain unchanged since invalid_status is not allowed
        assert result[0].status == "new"


class TestActionItemWorkerExecute:

    def test_execute_calls_dispatcher(self):
        response = json.dumps({"new_items": [{"description": "Deploy app"}], "updated_items": []})
        dispatcher = make_dispatcher(response)
        worker = ActionItemWorker(dispatcher)

        result = asyncio.get_event_loop().run_until_complete(
            worker.execute(full_context="ctx", recent_transcript="transcript", existing_items=[])
        )

        dispatcher.run.assert_called_once_with(
            "action_items",
            full_context="ctx",
            recent_transcript="transcript",
            existing_items="[]",
        )
        assert len(result) == 1
        assert result[0].description == "Deploy app"


# ---------------------------------------------------------------------------
# ContradictionWorker
# ---------------------------------------------------------------------------

class TestContradictionWorkerExtractJson:

    def test_plain_json(self):
        raw = '{"contradictions": []}'
        result = ContradictionWorker._extract_json(raw)
        assert result == {"contradictions": []}

    def test_fenced_json(self):
        raw = '```json\n{"contradictions": []}\n```'
        result = ContradictionWorker._extract_json(raw)
        assert result == {"contradictions": []}


class TestContradictionWorkerParseResponse:

    def _make_worker(self):
        return ContradictionWorker(make_dispatcher())

    def test_parses_contradictions(self):
        worker = self._make_worker()
        raw = json.dumps({
            "contradictions": [
                {
                    "description": "Budget conflict",
                    "statement_a": "Budget is $100k",
                    "statement_b": "Budget is $200k",
                    "severity": "high",
                }
            ]
        })
        result = worker._parse_response(raw)
        assert len(result) == 1
        assert result[0].description == "Budget conflict"
        assert result[0].severity == "high"

    def test_invalid_severity_defaults_to_low(self):
        worker = self._make_worker()
        raw = json.dumps({
            "contradictions": [{"description": "Conflict", "severity": "extreme"}]
        })
        result = worker._parse_response(raw)
        assert result[0].severity == "low"

    def test_invalid_json_returns_empty(self):
        worker = self._make_worker()
        result = worker._parse_response("garbage")
        assert result == []

    def test_skips_item_without_description(self):
        worker = self._make_worker()
        raw = json.dumps({"contradictions": [{"severity": "high"}]})
        result = worker._parse_response(raw)
        assert result == []


class TestContradictionWorkerExecute:

    def test_skips_empty_transcript(self):
        dispatcher = make_dispatcher()
        worker = ContradictionWorker(dispatcher)
        result = asyncio.get_event_loop().run_until_complete(
            worker.execute(current_summary="summary", recent_transcript="   ")
        )
        dispatcher.run.assert_not_called()
        assert result == []

    def test_execute_calls_dispatcher_when_transcript_present(self):
        response = json.dumps({"contradictions": []})
        dispatcher = make_dispatcher(response)
        worker = ContradictionWorker(dispatcher)

        result = asyncio.get_event_loop().run_until_complete(
            worker.execute(current_summary="sum", recent_transcript="Alice said X.")
        )

        dispatcher.run.assert_called_once()
        assert result == []


# ---------------------------------------------------------------------------
# ReplyWorker
# ---------------------------------------------------------------------------

class TestReplyWorkerParseResponse:

    def _make_worker(self):
        return ReplyWorker(make_dispatcher())

    def test_parses_suggestions(self):
        worker = self._make_worker()
        raw = json.dumps({"suggestions": ["Sure!", "Let me check."], "context": "deployment"})
        result = worker._parse_response(raw, "context_hint")
        assert isinstance(result, ReplySuggestion)
        assert result.suggestions == ["Sure!", "Let me check."]
        assert result.context == "deployment"

    def test_invalid_json_falls_back_to_raw(self):
        worker = self._make_worker()
        result = worker._parse_response("Plain text reply", "hint")
        assert result.suggestions == ["Plain text reply"]
        assert result.context == "hint"

    def test_filters_empty_suggestions(self):
        worker = self._make_worker()
        raw = json.dumps({"suggestions": ["Valid", "", "  "], "context": ""})
        result = worker._parse_response(raw, "")
        assert result.suggestions == ["Valid"]

    def test_non_list_suggestions_defaults_to_empty(self):
        worker = self._make_worker()
        raw = json.dumps({"suggestions": "not a list", "context": ""})
        result = worker._parse_response(raw, "")
        assert result.suggestions == []


class TestReplyWorkerExecute:

    def test_execute_returns_reply_suggestion(self):
        response = json.dumps({"suggestions": ["Sounds good"], "context": "ctx"})
        dispatcher = make_dispatcher(response)
        worker = ReplyWorker(dispatcher)

        result = asyncio.get_event_loop().run_until_complete(
            worker.execute(full_context="meeting ctx", context_hint="deployment question")
        )

        assert isinstance(result, ReplySuggestion)
        assert result.suggestions == ["Sounds good"]
        assert result.triggered_by == "manual"


# ---------------------------------------------------------------------------
# SummaryWorker
# ---------------------------------------------------------------------------

class TestSummaryWorkerExecute:

    def test_returns_current_summary_when_no_new_segments(self):
        dispatcher = make_dispatcher("ignored")
        worker = SummaryWorker(dispatcher)

        result = asyncio.get_event_loop().run_until_complete(
            worker.execute(current_summary="existing summary", new_segments="   ")
        )

        dispatcher.run.assert_not_called()
        assert result == "existing summary"

    def test_calls_dispatcher_and_returns_stripped_result(self):
        dispatcher = make_dispatcher("  Updated summary.  ")
        worker = SummaryWorker(dispatcher)

        result = asyncio.get_event_loop().run_until_complete(
            worker.execute(current_summary="old", new_segments="Alice: new stuff.")
        )

        dispatcher.run.assert_called_once()
        assert result == "Updated summary."

    def test_uses_default_placeholder_when_no_summary(self):
        dispatcher = make_dispatcher("First summary.")
        worker = SummaryWorker(dispatcher)

        asyncio.get_event_loop().run_until_complete(
            worker.execute(current_summary="", new_segments="Alice: Hello.")
        )

        call_kwargs = dispatcher.run.call_args[1]
        assert "first update" in call_kwargs["current_summary"].lower()


# ---------------------------------------------------------------------------
# CustomPromptWorker
# ---------------------------------------------------------------------------

class TestCustomPromptWorkerExecute:

    def test_returns_custom_prompt_result(self):
        dispatcher = make_dispatcher("  The answer.  ")
        worker = CustomPromptWorker(dispatcher)

        result = asyncio.get_event_loop().run_until_complete(
            worker.execute(
                full_context="meeting ctx",
                user_prompt="What was decided?",
                timestamp=42.0,
            )
        )

        assert isinstance(result, CustomPromptResult)
        assert result.result == "The answer."
        assert result.prompt == "What was decided?"
        assert result.timestamp == 42.0

    def test_uses_placeholder_when_no_context(self):
        dispatcher = make_dispatcher("answer")
        worker = CustomPromptWorker(dispatcher)

        asyncio.get_event_loop().run_until_complete(
            worker.execute(full_context="", user_prompt="Q?")
        )

        call_kwargs = dispatcher.run.call_args[1]
        assert "No meeting context" in call_kwargs["full_context"]

    def test_default_timestamp_is_zero(self):
        dispatcher = make_dispatcher("answer")
        worker = CustomPromptWorker(dispatcher)

        result = asyncio.get_event_loop().run_until_complete(
            worker.execute(full_context="ctx", user_prompt="Q?")
        )

        assert result.timestamp == 0.0
