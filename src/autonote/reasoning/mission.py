"""MissionBrief — pre-meeting briefing for coach mode.

Users create a YAML profile file describing their meeting goal, role,
available arguments, and coaching instructions. The ContextManager loads
this at startup and passes it to CoachWorker for proactive suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
        coach_every_n_turns: Auto-trigger interval (default: every 3 turns).
    """

    name: str
    goal: str
    role: str
    context: str = ""
    arguments: list[str] = field(default_factory=list)
    instructions: str = ""
    coach_every_n_turns: int = 3

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

        return cls(
            name=data["name"],
            goal=data["goal"],
            role=data["role"],
            context=data.get("context", ""),
            arguments=data.get("arguments", []),
            instructions=data.get("instructions", ""),
            coach_every_n_turns=int(data.get("coach_every_n_turns", 3)),
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
