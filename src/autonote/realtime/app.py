"""Textual TUI for the real-time meeting copilot.

Split-screen terminal dashboard:
  Left  — live scrolling transcript (Me / Them)
  Right — rolling summary, action items, contradiction alerts
  Bottom — input bar for custom prompts

Orchestrates: Recorder → Transcriber → ContextManager → TUI updates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, RichLog, Static

from autonote.realtime.models import (
    ActionItemsUpdate,
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
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(highlight=True, markup=True, wrap=True, **kwargs)
        self.border_title = "Transcript"


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
        save_recordings: Whether to save WAV files.
        title: Meeting title for metadata.
    """

    TITLE = "Autonote Realtime"
    SUB_TITLE = "Live Meeting Copilot"

    DEFAULT_CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 2fr 1fr;
        grid-rows: 1fr auto;
    }

    #transcript-container {
        row-span: 1;
    }

    #reasoning-container {
        row-span: 1;
    }

    #input-bar {
        column-span: 2;
        dock: bottom;
        height: 3;
    }

    Input {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "request_reply", "Reply Suggestions", show=True),
        Binding("ctrl+c", "quit", "Stop Recording", priority=True, show=False),
    ]

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        save_recordings: bool = False,
        title: str = "",
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._model = model
        self._save_recordings = save_recordings
        self._meeting_title = title

        # Pipeline objects — created in on_mount
        self._recorder = None
        self._transcriber = None
        self._context_manager = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield TranscriptLog(id="transcript")
            with VerticalScroll(id="reasoning-container"):
                yield SummaryPanel(id="summary")
                yield ActionItemsPanel(id="action-items")
                yield AlertsPanel(id="alerts")
        yield Input(placeholder="Ask a question about the meeting... (Enter to send)", id="prompt-input")
        yield StatusLine(id="status")
        yield Footer()

    async def on_mount(self) -> None:
        """Start the recording pipeline when the app mounts."""
        self._start_pipeline()

    @work(exclusive=True, thread=False)
    async def _start_pipeline(self) -> None:
        """Initialize and start the full pipeline: recorder → transcriber → context_manager."""
        from autonote.realtime.recorder import RealtimeRecorder
        from autonote.realtime.transcriber import RealtimeTranscriber as AAITranscriber
        from autonote.reasoning.context_manager import ContextManager
        from autonote.reasoning.dispatcher import LLMDispatcher

        status = self.query_one("#status", StatusLine)
        transcript_log = self.query_one("#transcript", TranscriptLog)

        try:
            # 1. Check system dependencies
            status.update("Checking dependencies...")
            ok, errors = await RealtimeRecorder.check_dependencies()
            if not ok:
                status.update(f"[red]Missing dependencies: {'; '.join(errors)}[/red]")
                return

            # 2. Start recorder
            status.update("Starting audio capture...")
            self._recorder = RealtimeRecorder(save_to_file=self._save_recordings)
            await self._recorder.start(title=self._meeting_title)

            mode = "mic + system" if self._recorder.has_monitor else "mic only"
            transcript_log.write(
                Text(f"Recording started ({mode}). Connecting to AssemblyAI...", style="dim italic")
            )

            # 3. Start transcriber
            status.update("Connecting to AssemblyAI...")
            monitor_q = self._recorder.monitor_queue if self._recorder.has_monitor else None
            self._transcriber = AAITranscriber(
                mic_queue=self._recorder.mic_queue,
                monitor_queue=monitor_q,
                api_key=self._api_key,
            )
            await self._transcriber.start()

            # 4. Set up reasoning engine
            dispatcher = LLMDispatcher(model=self._model)
            self._context_manager = ContextManager(
                dispatcher=dispatcher,
                on_event=self._handle_event,
            )

            transcript_log.write(
                Text("Connected. Listening...\n", style="bold green")
            )
            status.update("Recording...")

            # 5. Start consumer loop and status updater
            self._consumer_task = asyncio.create_task(self._consume_segments())
            self._status_task = asyncio.create_task(self._update_status_loop())

        except Exception as exc:
            status.update(f"[red]Pipeline error: {exc}[/red]")
            logger.error("Pipeline startup failed: %s", exc, exc_info=True)

    async def _consume_segments(self) -> None:
        """Read TranscriptSegments from the transcriber and feed them to the ContextManager."""
        assert self._transcriber is not None
        assert self._context_manager is not None

        while True:
            segment = await self._transcriber.segment_queue.get()
            if segment is None:
                break
            # Skip partials for context manager, but still display them
            if segment.is_partial:
                await self._handle_event(segment)
            else:
                await self._context_manager.on_new_segment(segment)

    async def _handle_event(self, event: RealtimeEvent) -> None:
        """Route a realtime event to the appropriate TUI widget."""
        try:
            if isinstance(event, TranscriptSegment):
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

    def _update_transcript(self, segment: TranscriptSegment) -> None:
        log = self.query_one("#transcript", TranscriptLog)
        if segment.is_partial:
            # For partials, we could show inline but it gets noisy.
            # Just skip — finals will cover it.
            return
        speaker = segment.speaker
        style = "bold cyan" if speaker == "Me" else "bold magenta"
        ts = f"{segment.timestamp_start:.0f}s"
        text = Text()
        text.append(f"[{ts}] ", style="dim")
        text.append(f"{speaker}: ", style=style)
        text.append(segment.text)
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
        while True:
            await asyncio.sleep(1.0)
            if self._recorder and self._recorder.is_recording:
                stats = self._recorder.recording_stats
                mins = int(stats.duration_seconds) // 60
                secs = int(stats.duration_seconds) % 60
                mb = stats.bytes_read / (1024 * 1024)
                segments = (
                    len(self._context_manager.state.segments)
                    if self._context_manager
                    else 0
                )
                self.query_one("#status", StatusLine).update(
                    f"Recording {mins:02d}:{secs:02d} | "
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

    async def action_request_reply(self) -> None:
        """Handle the 'r' key — request reply suggestions."""
        if self._context_manager:
            asyncio.create_task(self._context_manager.handle_reply_request())

    async def action_quit(self) -> None:
        """Graceful shutdown."""
        status = self.query_one("#status", StatusLine)
        status.update("Shutting down...")

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
    save_recordings: bool = False,
    title: str = "",
) -> None:
    """Entry point for ``autonote realtime``."""
    app = RealtimeApp(
        api_key=api_key,
        model=model,
        save_recordings=save_recordings,
        title=title,
    )
    app.run()
