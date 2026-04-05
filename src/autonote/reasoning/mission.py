"""MissionBrief — pre-meeting briefing for coach mode.

Users create a YAML profile file describing their meeting goal, role,
available arguments, and coaching instructions. The ContextManager loads
this at startup and passes it to CoachWorker for proactive suggestions.

Mode configuration (all optional, sensible defaults):
  summary_enabled / action_items_enabled / contradictions_enabled — toggle
    auto-triggers for those modes (default: True).
  reply_every_n_turns — auto-trigger reply suggestions every N turns
    (default: 0 = manual only).
  coach_every_n_turns — auto-trigger coach every N turns (default: 3).
  summary_every_n_turns / action_items_every_n_turns /
    contradictions_every_seconds — override default thresholds.

Panel visibility (all default True except coach, which requires a profile):
  panels.summary / panels.action_items / panels.alerts / panels.coach /
  panels.debug — show or hide each panel on startup.

Aggregator tuning:
  silence_threshold — seconds of silence before flushing a turn (default 2.0).
  max_turn_duration — max seconds before force-flushing a turn (default 15.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PanelConfig:
    """Visibility and sizing settings for each TUI panel.

    max_height values are CSS strings (e.g. "50%", "30vh").
    None means use the widget's default CSS.
    """

    summary: bool = True
    action_items: bool = True
    alerts: bool = True
    coach: bool = True
    debug: bool = True

    summary_max_height: str | None = None
    action_items_max_height: str | None = None
    alerts_max_height: str | None = None
    coach_max_height: str | None = None


@dataclass
class MissionBrief:
    """Structured briefing that shapes coach-mode LLM suggestions.

    Attributes:
        name: Human-readable profile name (e.g. "Contract Negotiation").
        goal: What the user is trying to achieve in this meeting.
        role: The user's role or persona (e.g. "Lead negotiator for Acme Corp").
        context: Background information about the meeting / counterparty.
        arguments: List of prepared talking points the coach can suggest.
        instructions: Extra coaching instructions for the LLM.

        summary_enabled: Whether to auto-run summary updates.
        action_items_enabled: Whether to auto-run action item scans.
        contradictions_enabled: Whether to auto-run contradiction checks.
        summary_every_n_turns: Trigger summary every N turns.
        action_items_every_n_turns: Trigger action scan every N turns.
        contradictions_every_seconds: Trigger contradiction check every N seconds.
        reply_every_n_turns: Auto-trigger reply suggestions every N turns (0 = off).
        coach_every_n_turns: Auto-trigger coach every N turns (0 = off).

        panels: Which TUI panels to show on startup.
        silence_threshold: Seconds of silence before flushing a turn.
        max_turn_duration: Max turn duration in seconds before force-flush.
    """

    name: str
    goal: str
    role: str
    context: str = ""
    arguments: list[str] = field(default_factory=list)
    instructions: str = ""

    # Mode toggles
    summary_enabled: bool = True
    action_items_enabled: bool = True
    contradictions_enabled: bool = True

    # Turn/time thresholds
    summary_every_n_turns: int = 5
    action_items_every_n_turns: int = 3
    contradictions_every_seconds: int = 120
    reply_every_n_turns: int = 0  # 0 = manual only
    coach_every_n_turns: int = 3

    # Panel visibility
    panels: PanelConfig = field(default_factory=PanelConfig)

    # Aggregator tuning
    silence_threshold: float = 2.0
    max_turn_duration: float = 15.0

    @classmethod
    def from_yaml(cls, path: str) -> MissionBrief:
        """Load a MissionBrief from a YAML file.

        Args:
            path: Path to the YAML profile file.

        Returns:
            Populated MissionBrief instance.

        Raises:
            FileNotFoundError: If the path does not exist.
            KeyError: If required fields (name, goal, role) are missing.
        """
        import yaml  # lazy import — only needed at profile load time

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        panels_data = data.get("panels", {})
        panels = PanelConfig(
            summary=panels_data.get("summary", True),
            action_items=panels_data.get("action_items", True),
            alerts=panels_data.get("alerts", True),
            coach=panels_data.get("coach", True),
            debug=panels_data.get("debug", True),
            summary_max_height=panels_data.get("summary_max_height"),
            action_items_max_height=panels_data.get("action_items_max_height"),
            alerts_max_height=panels_data.get("alerts_max_height"),
            coach_max_height=panels_data.get("coach_max_height"),
        )

        return cls(
            name=data["name"],
            goal=data["goal"],
            role=data["role"],
            context=data.get("context", ""),
            arguments=data.get("arguments", []),
            instructions=data.get("instructions", ""),
            summary_enabled=bool(data.get("summary_enabled", True)),
            action_items_enabled=bool(data.get("action_items_enabled", True)),
            contradictions_enabled=bool(data.get("contradictions_enabled", True)),
            summary_every_n_turns=int(data.get("summary_every_n_turns", 5)),
            action_items_every_n_turns=int(data.get("action_items_every_n_turns", 3)),
            contradictions_every_seconds=int(data.get("contradictions_every_seconds", 120)),
            reply_every_n_turns=int(data.get("reply_every_n_turns", 0)),
            coach_every_n_turns=int(data.get("coach_every_n_turns", 3)),
            panels=panels,
            silence_threshold=float(data.get("silence_threshold", 2.0)),
            max_turn_duration=float(data.get("max_turn_duration", 15.0)),
        )

    def format_for_prompt(self) -> str:
        """Format the brief as a structured text block for prompt injection."""
        args_text = "\n".join(f"- {a}" for a in self.arguments) if self.arguments else "(none)"
        parts = [
            f"Name: {self.name}",
            f"Your role: {self.role}",
            f"Goal: {self.goal}",
        ]
        if self.context:
            parts.append(f"Context: {self.context}")
        parts.append(f"Available arguments:\n{args_text}")
        if self.instructions:
            parts.append(f"Instructions: {self.instructions}")
        return "\n".join(parts)
