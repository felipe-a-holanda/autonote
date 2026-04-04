"""Tests for MissionBrief, CoachWorker, and coach integration in ContextManager."""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from autonote.reasoning.mission import MissionBrief
from autonote.reasoning.workers.coach import CoachWorker
from autonote.realtime.models import AggregatedTurn, CoachSuggestion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_brief(**overrides) -> MissionBrief:
    defaults = dict(
        name="Test Brief",
        goal="Win the deal",
        role="Lead negotiator",
        context="Negotiating a software contract.",
        arguments=["We have a competing offer", "5 years of loyalty"],
        instructions="Be concise.",
        coach_every_n_turns=3,
    )
    defaults.update(overrides)
    return MissionBrief(**defaults)


def make_dispatcher(return_value: str) -> MagicMock:
    dispatcher = MagicMock()
    dispatcher.run = AsyncMock(return_value=return_value)
    return dispatcher


def make_turn(text: str = "Let's talk price", speaker: str = "Them", start: float = 0.0, end: float = 2.0) -> AggregatedTurn:
    return AggregatedTurn(
        speaker=speaker,
        text=text,
        timestamp_start=start,
        timestamp_end=end,
        segment_count=1,
    )


# ---------------------------------------------------------------------------
# MissionBrief
# ---------------------------------------------------------------------------

class TestMissionBrief:

    def test_from_yaml(self, tmp_path):
        yaml_content = """
name: "Contract Negotiation"
goal: "Reduce cost by 20%"
role: "Lead negotiator"
context: "Renewing with VendorCo."
arguments:
  - "AltVendor offered $75k"
  - "5 years loyalty"
instructions: "Stay diplomatic."
coach_every_n_turns: 2
"""
        profile = tmp_path / "test.yaml"
        profile.write_text(yaml_content)

        brief = MissionBrief.from_yaml(str(profile))

        assert brief.name == "Contract Negotiation"
        assert brief.goal == "Reduce cost by 20%"
        assert brief.role == "Lead negotiator"
        assert "VendorCo" in brief.context
        assert len(brief.arguments) == 2
        assert brief.arguments[0] == "AltVendor offered $75k"
        assert brief.instructions == "Stay diplomatic."
        assert brief.coach_every_n_turns == 2

    def test_from_yaml_defaults(self, tmp_path):
        yaml_content = "name: Minimal\ngoal: Test\nrole: Tester\n"
        profile = tmp_path / "minimal.yaml"
        profile.write_text(yaml_content)

        brief = MissionBrief.from_yaml(str(profile))

        assert brief.context == ""
        assert brief.arguments == []
        assert brief.instructions == ""
        assert brief.coach_every_n_turns == 3

    def test_format_for_prompt_contains_key_fields(self):
        brief = make_brief()
        text = brief.format_for_prompt()

        assert "Win the deal" in text
        assert "Lead negotiator" in text
        assert "We have a competing offer" in text
        assert "5 years of loyalty" in text
        assert "Be concise" in text

    def test_format_for_prompt_no_arguments(self):
        brief = make_brief(arguments=[])
        text = brief.format_for_prompt()
        assert "(none)" in text


# ---------------------------------------------------------------------------
# CoachWorker
# ---------------------------------------------------------------------------

class TestCoachWorker:

    async def test_execute_parses_valid_json(self):
        payload = {
            "should_speak": True,
            "suggestion": "Mention our competing offer now.",
            "argument_used": "We have a competing offer",
            "reasoning": "They just raised price concerns.",
            "confidence": "high",
        }
        dispatcher = make_dispatcher(json.dumps(payload))
        worker = CoachWorker(dispatcher)

        result = await worker.execute(
            full_context="summary: ...",
            recent_transcript="Them: We can't go lower than $100k.",
            mission_brief_text="Goal: Reduce cost",
            timestamp=10.0,
        )

        assert isinstance(result, CoachSuggestion)
        assert result.should_speak is True
        assert result.suggestion == "Mention our competing offer now."
        assert result.argument_used == "We have a competing offer"
        assert result.confidence == "high"
        assert result.timestamp == 10.0

    async def test_execute_handles_invalid_json(self):
        dispatcher = make_dispatcher("This is not JSON at all.")
        worker = CoachWorker(dispatcher)

        result = await worker.execute(
            full_context="...",
            recent_transcript="...",
            mission_brief_text="...",
            timestamp=5.0,
        )

        assert isinstance(result, CoachSuggestion)
        assert result.should_speak is False
        assert result.confidence == "low"
        assert result.timestamp == 5.0

    async def test_execute_handles_json_fenced_in_markdown(self):
        payload = {"should_speak": False, "suggestion": "Wait.", "argument_used": None, "reasoning": "Too early.", "confidence": "medium"}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        dispatcher = make_dispatcher(fenced)
        worker = CoachWorker(dispatcher)

        result = await worker.execute(mission_brief_text="...", timestamp=0.0)

        assert result.should_speak is False
        assert result.confidence == "medium"
        assert result.argument_used is None

    async def test_execute_clamps_unknown_confidence(self):
        payload = {"should_speak": True, "suggestion": "Go!", "argument_used": None, "reasoning": "Now.", "confidence": "ultra-high"}
        dispatcher = make_dispatcher(json.dumps(payload))
        worker = CoachWorker(dispatcher)

        result = await worker.execute(timestamp=0.0)

        assert result.confidence == "low"  # unknown values fall back to "low"


# ---------------------------------------------------------------------------
# ContextManager coach integration
# ---------------------------------------------------------------------------

class TestContextManagerCoach:

    def _make_cm(self, mission_brief=None, coach_every_n_turns=None):
        from autonote.reasoning.context_manager import ContextManager
        dispatcher = make_dispatcher('{"new_items": [], "updated_items": []}')
        on_event = AsyncMock()
        cm = ContextManager(
            dispatcher=dispatcher,
            on_event=on_event,
            mission_brief=mission_brief,
        )
        if coach_every_n_turns is not None:
            cm.COACH_EVERY_N_TURNS = coach_every_n_turns
        return cm, on_event

    async def test_coach_fires_every_n_turns(self):
        import asyncio
        brief = make_brief(coach_every_n_turns=2)
        cm, on_event = self._make_cm(mission_brief=brief, coach_every_n_turns=2)

        # Mock the coach worker so we don't need a real LLM
        coach_payload = json.dumps({
            "should_speak": True,
            "suggestion": "Deploy argument now.",
            "argument_used": "Competing offer",
            "reasoning": "Price mentioned.",
            "confidence": "high",
        })
        cm._coach_worker.dispatcher.run = AsyncMock(return_value=coach_payload)

        for i in range(2):
            await cm.on_new_turn(make_turn(start=float(i), end=float(i + 1)))

        # Allow background tasks spawned by _fire_task to complete
        await asyncio.gather(*list(cm._running_tasks), return_exceptions=True)

        # After 2 turns at threshold=2, coach should have fired
        coach_events = [
            call.args[0] for call in on_event.call_args_list
            if isinstance(call.args[0], CoachSuggestion)
        ]
        assert len(coach_events) >= 1
        assert cm.state.turns_since_last_coach == 0  # reset after firing

    async def test_coach_does_not_fire_without_profile(self):
        cm, on_event = self._make_cm(mission_brief=None)

        for i in range(10):
            await cm.on_new_turn(make_turn(start=float(i), end=float(i + 1)))

        coach_events = [
            call.args[0] for call in on_event.call_args_list
            if isinstance(call.args[0], CoachSuggestion)
        ]
        assert len(coach_events) == 0

    async def test_handle_coach_request_manual(self):
        brief = make_brief()
        cm, on_event = self._make_cm(mission_brief=brief)

        coach_payload = json.dumps({
            "should_speak": False,
            "suggestion": "Wait.",
            "argument_used": None,
            "reasoning": "Not yet.",
            "confidence": "low",
        })
        cm._coach_worker.dispatcher.run = AsyncMock(return_value=coach_payload)
        cm.state.turns.append(make_turn())

        await cm.handle_coach_request()

        coach_events = [
            call.args[0] for call in on_event.call_args_list
            if isinstance(call.args[0], CoachSuggestion)
        ]
        assert len(coach_events) == 1

    async def test_handle_coach_request_no_op_without_profile(self):
        cm, on_event = self._make_cm(mission_brief=None)

        await cm.handle_coach_request()

        coach_events = [
            call.args[0] for call in on_event.call_args_list
            if isinstance(call.args[0], CoachSuggestion)
        ]
        assert len(coach_events) == 0
