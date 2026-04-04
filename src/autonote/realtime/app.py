"""Textual TUI for the real-time meeting copilot.

Split-screen terminal dashboard:
  Left  — live scrolling transcript (Me / Them)
  Right — rolling summary, action items, contradiction alerts
  Bottom — input bar for custom prompts

Orchestrates: Recorder → Transcriber → ContextManager → TUI updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, RichLog, Static

from autonote.realtime.models import (
    ActionItemsUpdate,
    AggregatedTurn,
    ContradictionAlert,
    CustomPromptResult,
    RealtimeEvent,
    ReplySuggestion,
    SummaryUpdate,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------

class TranscriptLog(RichLog):
    """Scrolling transcript view with speaker-colored entries."""

    DEFAULT_CSS = """
    TranscriptLog {
        border: solid $primary;
        border-title-color: $text;
        scrollbar-size: 1 1;
        height: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(highlight=True, markup=True, wrap=True, **kwargs)
        self.border_title = "Transcript"


class PartialLine(Static):
    """Live partial transcript line — updates in place as speech is recognized."""

    DEFAULT_CSS = """
    PartialLine {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)


class SummaryPanel(Static):
    """Displays the current progressive summary."""

    DEFAULT_CSS = """
    SummaryPanel {
        border: solid $accent;
        border-title-color: $text;
        padding: 1;
        height: auto;
        min-height: 5;
        max-height: 50%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("Waiting for transcript...", **kwargs)
        self.border_title = "Summary"


class ActionItemsPanel(Static):
    """Displays extracted action items."""

    DEFAULT_CSS = """
    ActionItemsPanel {
        border: solid $warning;
        border-title-color: $text;
        padding: 1;
        height: auto;
        min-height: 3;
        max-height: 30%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("No action items yet.", **kwargs)
        self.border_title = "Action Items"


class AlertsPanel(Static):
    """Displays contradiction alerts and reply suggestions."""

    DEFAULT_CSS = """
    AlertsPanel {
        border: solid $error;
        border-title-color: $text;
        padding: 1;
        height: auto;
        min-height: 3;
        max-height: 20%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.border_title = "Alerts"
        self.display = False  # Hidden until first alert


class DebugLog(RichLog):
    """Timestamped debug log for pipeline events."""

    DEFAULT_CSS = """
    DebugLog {
        border: solid $surface-lighten-2;
        border-title-color: $text-muted;
        scrollbar-size: 1 1;
        height: 1fr;
        min-height: 6;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(highlight=False, markup=True, wrap=True, **kwargs)
        self.border_title = "Debug  [dim](d=export)[/dim]"
        self._lines: list[str] = []

    def log(self, msg: str, level: str = "info") -> None:
        ts = time.strftime("%H:%M:%S")
        color = {"info": "dim", "ok": "green", "warn": "yellow", "error": "red"}.get(level, "dim")
        self._lines.append(f"[{ts}] {msg}")
        self.write(Text.from_markup(f"[dim]{ts}[/dim] [{color}]{msg}[/{color}]"))

    def export(self) -> str:
        return "\n".join(self._lines)


class StatusLine(Static):
    """Bottom status bar showing recording stats."""

    DEFAULT_CSS = """
    StatusLine {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    StatusLine.connecting {
        background: $warning 20%;
        color: $warning;
    }
    StatusLine.ready {
        background: $success 10%;
        color: $success;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("Initializing...", **kwargs)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class RealtimeApp(App):
    """Real-time meeting copilot TUI.

    Args:
        api_key: AssemblyAI API key (overrides config).
        model: LLM model/preset for reasoning (overrides config).
        title: Meeting title for metadata.
    """

    TITLE = "Autonote Realtime"
    SUB_TITLE = "Live Meeting Copilot"

    DEFAULT_CSS = """
    Screen {
        layout: grid;
        grid-size: 3 2;
        grid-columns: 2fr 1fr 1fr;
        grid-rows: 1fr auto;
    }

    #transcript-container {
        row-span: 1;
    }

    #reasoning-container {
        row-span: 1;
    }

    #debug {
        row-span: 1;
    }

    #input-bar {
        column-span: 3;
        dock: bottom;
        height: 3;
    }

    Input {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("s", "request_summary", "Summary", show=True),
        Binding("a", "request_action_items", "Action Items", show=True),
        Binding("c", "request_contradictions", "Contradictions", show=True),
        Binding("r", "request_reply", "Reply", show=True),
        Binding("d", "export_debug", "Export Debug", show=True),
        Binding("ctrl+c", "quit", "Stop Recording", priority=True, show=False),
    ]

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        title: str = "",
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._model = model
        self._meeting_title = title

        # Pipeline objects — created in on_mount
        self._recorder = None
        self._transcriber = None
        self._aggregator = None
        self._context_manager = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._bridge_task: Optional[asyncio.Task] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None
        self._transcript_path: Optional[Path] = None

        # Connection state tracking
        self._sessions_ready: set[str] = set()
        self._expected_sessions: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="transcript-container"):
                yield TranscriptLog(id="transcript")
                yield PartialLine(id="partial-line")
            with VerticalScroll(id="reasoning-container"):
                yield SummaryPanel(id="summary")
                yield ActionItemsPanel(id="action-items")
                yield AlertsPanel(id="alerts")
            yield DebugLog(id="debug")
        yield Input(placeholder="Ask a question about the meeting... (Enter to send)", id="prompt-input")
        yield StatusLine(id="status")
        yield Footer()

    async def on_mount(self) -> None:
        """Start the recording pipeline when the app mounts."""
        self._start_pipeline()

    def _debug(self, msg: str, level: str = "info") -> None:
        """Write a timestamped message to the debug panel."""
        try:
            self.query_one("#debug", DebugLog).log(msg, level)
        except Exception:
            pass

    def _on_session_begin(self, speaker_label: str) -> None:
        """Called (thread-safe) when an AssemblyAI session is fully open."""
        self._sessions_ready.add(speaker_label)
        status = self.query_one("#status", StatusLine)
        transcript_log = self.query_one("#transcript", TranscriptLog)

        if self._sessions_ready >= self._expected_sessions:
            # All expected sessions are open — switch to ready state
            status.remove_class("connecting")
            status.add_class("ready")
            status.update("● Ready — listening for speech  |  q=quit  r=reply suggestions")
            transcript_log.write(
                Text(f"[{time.strftime('%H:%M:%S')}] Connected. Listening...\n", style="bold green")
            )
            self._debug("All sessions ready — transcription active", "ok")
        else:
            missing = self._expected_sessions - self._sessions_ready
            status.update(f"⟳ Connecting... ({speaker_label} ready, waiting for {', '.join(missing)})")

    @work(exclusive=True, thread=False)
    async def _start_pipeline(self) -> None:
        """Initialize and start the full pipeline: recorder → transcriber → aggregator → context_manager."""
        from autonote.realtime.aggregator import TurnAggregator
        from autonote.realtime.recorder import RealtimeRecorder
        from autonote.realtime.transcriber import RealtimeTranscriber as AAITranscriber
        from autonote.reasoning.context_manager import ContextManager
        from autonote.reasoning.dispatcher import LLMDispatcher

        status = self.query_one("#status", StatusLine)
        transcript_log = self.query_one("#transcript", TranscriptLog)

        try:
            # 1. Check system dependencies
            self._debug("Checking system dependencies...")
            status.update("Checking dependencies...")
            ok, errors = await RealtimeRecorder.check_dependencies()
            if not ok:
                for err in errors:
                    self._debug(f"Missing dep: {err}", "error")
                status.update(f"[red]Missing dependencies: {'; '.join(errors)}[/red]")
                return
            self._debug("Dependencies OK", "ok")

            # 2. Start recorder
            self._debug("Starting audio capture...")
            status.update("Starting audio capture...")
            self._recorder = RealtimeRecorder(save_to_file=True)
            await self._recorder.start(title=self._meeting_title)

            if self._recorder.meeting_dir:
                self._transcript_path = Path(self._recorder.meeting_dir) / "transcript.jsonl"
                self._debug(f"Transcript → {self._transcript_path}", "ok")

            mode = "mic + system" if self._recorder.has_monitor else "mic only"
            self._debug(f"Recorder started ({mode})", "ok")

            # 3. Start transcriber — this blocks while establishing WebSocket connections
            self._debug("Connecting to AssemblyAI WebSocket...")
            status.add_class("connecting")
            status.update("⟳ Connecting to AssemblyAI... (this may take ~20s)")
            transcript_log.write(
                Text(f"[{time.strftime('%H:%M:%S')}] Establishing connection to AssemblyAI...", style="yellow italic")
            )

            monitor_q = self._recorder.monitor_queue if self._recorder.has_monitor else None
            self._expected_sessions = {"Me"} | ({"Them"} if monitor_q is not None else set())

            self._transcriber = AAITranscriber(
                mic_queue=self._recorder.mic_queue,
                monitor_queue=monitor_q,
                api_key=self._api_key,
                on_debug=self._debug,
                on_session_begin=self._on_session_begin,
            )
            await self._transcriber.start()
            self._debug("AssemblyAI WebSocket connected — waiting for session confirmation", "ok")

            # 4. Set up reasoning engine
            self._debug(f"Loading LLM dispatcher (model={self._model or 'default'})...")
            dispatcher = LLMDispatcher(model=self._model)
            self._context_manager = ContextManager(
                dispatcher=dispatcher,
                on_event=self._handle_event,
                on_debug=self._debug,
            )
            self._debug("Reasoning engine ready", "ok")

            # 5. Instantiate aggregator and start bridge + consumer tasks
            self._aggregator = TurnAggregator()
            self._bridge_task = asyncio.create_task(self._bridge_segments())
            self._consumer_task = asyncio.create_task(self._consume_segments())
            self._status_task = asyncio.create_task(self._update_status_loop())
            self._debug("Pipeline running — waiting for speech", "ok")

        except Exception as exc:
            self._debug(f"Pipeline error: {exc}", "error")
            status.update(f"[red]Pipeline error: {exc}[/red]")
            logger.error("Pipeline startup failed: %s", exc, exc_info=True)

    async def _bridge_segments(self) -> None:
        """Read TranscriptSegments from the transcriber and feed them into the aggregator."""
        assert self._transcriber is not None
        assert self._aggregator is not None

        while True:
            segment = await self._transcriber.segment_queue.get()
            if segment is None:
                self._debug("Transcriber segment queue closed", "warn")
                self._aggregator.flush_remaining()
                self._aggregator.output_queue.put_nowait(None)
                break
            self._aggregator.feed(segment)

    async def _consume_segments(self) -> None:
        """Read from aggregator output queue and feed events to the ContextManager."""
        assert self._aggregator is not None
        assert self._context_manager is not None

        turn_count = 0
        while True:
            item = await self._aggregator.output_queue.get()
            if item is None:
                self._debug("Aggregator output queue closed", "warn")
                break
            if isinstance(item, TranscriptSegment):
                # Partial — display only, skip context manager
                await self._handle_event(item)
            elif isinstance(item, AggregatedTurn):
                turn_count += 1
                self._debug(f"Turn #{turn_count} [{item.speaker}]: {item.text[:60]}")
                self._append_transcript(item)
                await self._context_manager.on_new_turn(item)

    def _append_transcript(self, segment: TranscriptSegment | AggregatedTurn) -> None:
        """Append a transcript entry as a JSON line to the transcript file."""
        if self._transcript_path is None:
            return
        try:
            record = {
                "speaker": segment.speaker,
                "text": segment.text,
                "start": round(segment.timestamp_start, 3),
                "end": round(segment.timestamp_end, 3),
                "wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if isinstance(segment, AggregatedTurn):
                record["segment_count"] = segment.segment_count
            with self._transcript_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning("Failed to write transcript line: %s", exc)

    async def _handle_event(self, event: RealtimeEvent) -> None:
        """Route a realtime event to the appropriate TUI widget."""
        try:
            if isinstance(event, AggregatedTurn):
                self._update_transcript_turn(event)
            elif isinstance(event, TranscriptSegment):
                self._update_transcript(event)
            elif isinstance(event, SummaryUpdate):
                self._update_summary(event)
            elif isinstance(event, ActionItemsUpdate):
                self._update_action_items(event)
            elif isinstance(event, ContradictionAlert):
                self._update_alerts_contradiction(event)
            elif isinstance(event, ReplySuggestion):
                self._update_alerts_reply(event)
            elif isinstance(event, CustomPromptResult):
                self._update_alerts_custom(event)
        except Exception as exc:
            logger.warning("Error updating TUI for event %s: %s", type(event).__name__, exc)

    # ------------------------------------------------------------------
    # Widget updaters
    # ------------------------------------------------------------------

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format a float timestamp (seconds) as M:SS."""
        total = int(seconds)
        return f"{total // 60}:{total % 60:02d}"

    @staticmethod
    def _speaker_style(speaker: str) -> str:
        return "bold cyan" if speaker == "Me" else "bold magenta"

    def _update_partial_line(self, segment: TranscriptSegment) -> None:
        """Update the live partial line widget with the current partial text."""
        try:
            partial_line = self.query_one("#partial-line", PartialLine)
            text = Text()
            text.append(f"{segment.speaker}: ", style=f"{self._speaker_style(segment.speaker)} dim italic")
            text.append(segment.text, style="dim italic")
            partial_line.update(text)
        except Exception:
            pass

    def _clear_partial_line(self) -> None:
        """Clear the live partial line (called when a completed turn is written)."""
        try:
            self.query_one("#partial-line", PartialLine).update("")
        except Exception:
            pass

    def _update_transcript(self, segment: TranscriptSegment) -> None:
        if segment.is_partial:
            self._update_partial_line(segment)
            return
        log = self.query_one("#transcript", TranscriptLog)
        speaker = segment.speaker
        ts = self._format_timestamp(segment.timestamp_start)
        text = Text()
        text.append(f"[{ts}] ", style="dim")
        text.append(f"{speaker}: ", style=self._speaker_style(speaker))
        text.append(segment.text)
        log.write(text)

    def _update_transcript_turn(self, turn: AggregatedTurn) -> None:
        self._clear_partial_line()
        log = self.query_one("#transcript", TranscriptLog)
        speaker = turn.speaker
        ts = self._format_timestamp(turn.timestamp_start)
        text = Text()
        text.append(f"[{ts}] ", style="dim")
        text.append(f"{speaker}: ", style=self._speaker_style(speaker))
        text.append(turn.text)
        log.write(text)

    def _update_summary(self, update: SummaryUpdate) -> None:
        panel = self.query_one("#summary", SummaryPanel)
        panel.update(update.summary)

    def _update_action_items(self, update: ActionItemsUpdate) -> None:
        panel = self.query_one("#action-items", ActionItemsPanel)
        if not update.items:
            panel.update("No action items yet.")
            return
        lines = []
        for item in update.items:
            status_icon = {"new": "+", "updated": "~", "completed": "v"}.get(item.status, "?")
            assignee = f" @{item.assignee}" if item.assignee else ""
            lines.append(f"[{status_icon}] {item.description}{assignee}")
        panel.update("\n".join(lines))

    def _update_alerts_contradiction(self, alert: ContradictionAlert) -> None:
        panel = self.query_one("#alerts", AlertsPanel)
        panel.display = True
        severity_color = {"low": "yellow", "medium": "dark_orange", "high": "red"}.get(
            alert.severity, "yellow"
        )
        current = str(panel.renderable) if panel.renderable else ""
        new_text = (
            f"[{severity_color}]CONTRADICTION ({alert.severity}):[/{severity_color}] "
            f"{alert.description}\n"
        )
        panel.update(current + new_text if current else new_text)

    def _update_alerts_reply(self, suggestion: ReplySuggestion) -> None:
        panel = self.query_one("#alerts", AlertsPanel)
        panel.display = True
        lines = ["[bold]Reply Suggestions:[/bold]"]
        for i, s in enumerate(suggestion.suggestions, 1):
            lines.append(f"  {i}. {s}")
        if suggestion.context:
            lines.append(f"  [dim]({suggestion.context})[/dim]")
        panel.update("\n".join(lines))
        panel.border_title = "Reply Suggestions"

    def _update_alerts_custom(self, result: CustomPromptResult) -> None:
        panel = self.query_one("#alerts", AlertsPanel)
        panel.display = True
        panel.update(f"[bold]Q:[/bold] {result.prompt}\n\n{result.result}")
        panel.border_title = "Custom Prompt Result"

    # ------------------------------------------------------------------
    # Status updater
    # ------------------------------------------------------------------

    async def _update_status_loop(self) -> None:
        """Periodically update the status bar with recording stats."""
        first_tick = True
        while True:
            await asyncio.sleep(1.0)
            if self._recorder and self._recorder.is_recording:
                # Only show recording stats once all sessions are ready
                if self._sessions_ready < self._expected_sessions:
                    continue
                stats = self._recorder.recording_stats
                mins = int(stats.duration_seconds) // 60
                secs = int(stats.duration_seconds) % 60
                mb = stats.bytes_read / (1024 * 1024)
                segments = (
                    len(self._context_manager.state.segments)
                    if self._context_manager
                    else 0
                )
                status = self.query_one("#status", StatusLine)
                if first_tick:
                    status.remove_class("ready")
                    first_tick = False
                status.update(
                    f"● Recording {mins:02d}:{secs:02d} | "
                    f"{mb:.1f} MB captured | "
                    f"{segments} segments | "
                    f"q=quit  r=reply suggestions"
                )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @on(Input.Submitted, "#prompt-input")
    async def on_prompt_submitted(self, event: Input.Submitted) -> None:
        """Handle custom prompt submission."""
        prompt = event.value.strip()
        if not prompt or not self._context_manager:
            return
        event.input.value = ""
        # Show the prompt in the transcript log
        log = self.query_one("#transcript", TranscriptLog)
        log.write(Text(f"[You asked]: {prompt}", style="bold yellow"))
        # Fire the reasoning task
        asyncio.create_task(self._context_manager.handle_custom_prompt(prompt))

    async def action_request_summary(self) -> None:
        if self._context_manager:
            asyncio.create_task(self._context_manager.handle_summary_request())

    async def action_request_action_items(self) -> None:
        if self._context_manager:
            asyncio.create_task(self._context_manager.handle_action_items_request())

    async def action_request_contradictions(self) -> None:
        if self._context_manager:
            asyncio.create_task(self._context_manager.handle_contradiction_request())

    async def action_request_reply(self) -> None:
        """Handle the 'r' key — request reply suggestions."""
        if self._context_manager:
            asyncio.create_task(self._context_manager.handle_reply_request())

    async def action_export_debug(self) -> None:
        """Handle the 'd' key — export debug log to a timestamped file."""
        import pathlib
        debug_log = self.query_one("#debug", DebugLog)
        content = debug_log.export()
        if not content:
            self._debug("Nothing to export yet.", "warn")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = pathlib.Path(f"autonote_debug_{ts}.txt")
        path.write_text(content)
        self._debug(f"Exported to {path.resolve()}", "ok")

    async def action_quit(self) -> None:
        """Graceful shutdown."""
        status = self.query_one("#status", StatusLine)
        status.update("Shutting down...")

        if self._aggregator:
            self._aggregator.flush_remaining()
        if self._bridge_task and not self._bridge_task.done():
            self._bridge_task.cancel()
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
        if self._status_task and not self._status_task.done():
            self._status_task.cancel()
        if self._context_manager:
            await self._context_manager.shutdown()
        if self._transcriber:
            await self._transcriber.stop()
        if self._recorder and self._recorder.is_recording:
            stats = await self._recorder.stop()
            logger.info("Final recording stats: %s", stats)

        self.exit()


def run_realtime_app(
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    title: str = "",
) -> None:
    """Entry point for ``autonote realtime``."""
    app = RealtimeApp(
        api_key=api_key,
        model=model,
        title=title,
    )
    app.run()
