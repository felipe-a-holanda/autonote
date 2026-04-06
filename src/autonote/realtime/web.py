"""FastAPI web server for the real-time meeting copilot.

Provides the same pipeline as the Textual TUI (Recorder → Transcriber →
Aggregator → ContextManager) but streams events over WebSocket to a
browser-based frontend with scrollable, persistent panels.

Usage:
    autonote realtime-web [--profile YAML] [--port 8765] [--host 127.0.0.1]
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from autonote.realtime.event_bus import EventBus
from autonote.realtime.models import (
    ActionItemsUpdate,
    AggregatedTurn,
    RealtimeEvent,
    SummaryUpdate,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level pipeline state (lives for the server's lifetime)
# ---------------------------------------------------------------------------
_event_bus: EventBus = EventBus()
_context_manager = None  # Set during lifespan
_recorder = None
_transcriber = None
_aggregator = None
_pipeline_tasks: list[asyncio.Task] = []
_transcript_path: Optional[Path] = None
_transcript_txt_path: Optional[Path] = None

# Configuration injected before server starts
_api_key: Optional[str] = None
_model: Optional[str] = None
_meeting_title: str = ""
_profile = None  # MissionBrief or None
_full_transcript: bool = False

STATIC_DIR = Path(__file__).parent / "static"


def _debug(msg: str, level: str = "info") -> None:
    logger.debug("[web] %s", msg)


async def _bridge_segments() -> None:
    """Read TranscriptSegments from the transcriber and feed them into the aggregator."""
    while True:
        segment = await _transcriber.segment_queue.get()
        if segment is None:
            _aggregator.flush_remaining()
            _aggregator.output_queue.put_nowait(None)
            break
        _aggregator.feed(segment)


async def _consume_segments() -> None:
    """Read from aggregator output and feed to ContextManager + transcript file."""
    global _transcript_path
    turn_count = 0
    while True:
        item = await _aggregator.output_queue.get()
        if item is None:
            break
        if isinstance(item, TranscriptSegment):
            await _event_bus.publish(item)
        elif isinstance(item, AggregatedTurn):
            turn_count += 1
            _append_transcript(item)
            logger.info("[web] [→UI] %s: \"%s\"", item.speaker, item.text)
            await _context_manager.on_new_turn(item)


def _append_transcript(segment: TranscriptSegment | AggregatedTurn) -> None:
    """Append a transcript entry as JSONL and (for complete turns) as plain text."""
    if _transcript_path is None:
        return
    try:
        record = {
            "speaker": segment.speaker,
            "text": segment.text,
            "start": round(segment.timestamp_start, 3),
            "end": round(segment.timestamp_end, 3),
        }
        if isinstance(segment, AggregatedTurn):
            record["segment_count"] = segment.segment_count
            if segment.wall_time_start:
                record["wall_time"] = segment.wall_time_start.isoformat()
        with open(_transcript_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed to write transcript line: %s", exc)
        return
    if _transcript_txt_path is not None and isinstance(segment, AggregatedTurn):
        try:
            import time
            ts = time.strftime("%H:%M:%S", time.gmtime())
            with open(_transcript_txt_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {segment.speaker}: {segment.text}\n")
        except Exception as exc:
            logger.warning("Failed to write transcript txt line: %s", exc)


async def _start_pipeline() -> None:
    """Initialize and start the full pipeline."""
    global _context_manager, _recorder, _transcriber, _aggregator, _transcript_path, _transcript_txt_path

    from autonote.realtime.aggregator import TurnAggregator
    from autonote.realtime.recorder import RealtimeRecorder
    from autonote.realtime.transcriber import RealtimeTranscriber as AAITranscriber
    from autonote.reasoning.context_manager import ContextManager
    from autonote.reasoning.dispatcher import LLMDispatcher

    # 1. Check dependencies
    _debug("Checking system dependencies...")
    ok, errors = await RealtimeRecorder.check_dependencies()
    if not ok:
        raise RuntimeError(f"Missing dependencies: {'; '.join(errors)}")

    # 2. Start recorder
    _debug("Starting audio capture...")
    _recorder = RealtimeRecorder(save_to_file=True)
    await _recorder.start(title=_meeting_title)
    if _recorder.meeting_dir:
        _transcript_path = Path(_recorder.meeting_dir) / "transcript.jsonl"
        _transcript_txt_path = Path(_recorder.meeting_dir) / "transcript.txt"
    mode = "mic + system" if _recorder.has_monitor else "mic only"
    _debug(f"Recorder started ({mode})")

    # 3. Start transcriber
    _debug("Connecting to AssemblyAI...")
    monitor_q = _recorder.monitor_queue if _recorder.has_monitor else None
    _transcriber = AAITranscriber(
        mic_queue=_recorder.mic_queue,
        monitor_queue=monitor_q,
        api_key=_api_key,
        on_debug=_debug,
    )
    await _transcriber.start()
    _debug("AssemblyAI connected")

    # 4. Reasoning engine
    _debug(f"Loading LLM dispatcher (model={_model or 'default'})...")
    dispatcher = LLMDispatcher(model=_model)
    _context_manager = ContextManager(
        dispatcher=dispatcher,
        on_event=_event_bus.publish,
        on_debug=_debug,
        mission_brief=_profile,
        full_transcript=_full_transcript,
    )
    _debug("Reasoning engine ready")

    # 5. Aggregator + background tasks
    agg_kwargs: dict = {}
    if _profile is not None:
        agg_kwargs["silence_threshold"] = _profile.silence_threshold
        agg_kwargs["max_turn_duration"] = _profile.max_turn_duration
    _aggregator = TurnAggregator(**agg_kwargs, on_debug=_debug)

    _pipeline_tasks.append(asyncio.create_task(_bridge_segments()))
    _pipeline_tasks.append(asyncio.create_task(_consume_segments()))
    _pipeline_tasks.append(asyncio.create_task(_aggregator.run_silence_timer()))
    _debug("Pipeline running — waiting for speech")


async def _stop_pipeline() -> None:
    """Graceful shutdown of the pipeline."""
    if _aggregator:
        _aggregator.flush_remaining()
    for task in _pipeline_tasks:
        if not task.done():
            task.cancel()
    if _context_manager:
        await _context_manager.shutdown()
    if _transcriber:
        await _transcriber.stop()
    if _recorder and _recorder.is_recording:
        await _recorder.stop()
    _debug("Pipeline stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await _start_pipeline()
    yield
    await _stop_pipeline()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/config")
async def get_config():
    """Return panel visibility and layout config derived from the loaded profile."""
    if _profile is not None:
        p = _profile.panels
        return {
            "panels": {
                "summary": p.summary,
                "action_items": p.action_items,
                "alerts": p.alerts,
                "coach": p.coach,
            },
            "max_heights": {
                "summary": p.summary_max_height,
                "action_items": p.action_items_max_height,
                "alerts": p.alerts_max_height,
                "coach": p.coach_max_height,
            },
        }
    return {
        "panels": {"summary": True, "action_items": True, "alerts": True, "coach": False},
        "max_heights": {},
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)

    async def subscriber(event: RealtimeEvent) -> None:
        data = event.model_dump_json()
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            # Drop oldest to avoid backpressure stalling the pipeline
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(data)

    await _event_bus.subscribe(subscriber)
    try:
        # Send state snapshot for late-joining clients
        await _send_snapshot(ws)

        # Run send/receive concurrently
        send_task = asyncio.create_task(_ws_send(ws, queue))
        recv_task = asyncio.create_task(_ws_recv(ws))
        done, pending = await asyncio.wait(
            [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        await _event_bus.unsubscribe(subscriber)


async def _send_snapshot(ws: WebSocket) -> None:
    """Send current meeting state to a newly connected client."""
    if _context_manager is None:
        return
    state = _context_manager.state
    # Recent turns
    for turn in state.turns[-20:]:
        await ws.send_text(turn.model_dump_json())
    # Current summary
    if state.current_summary:
        await ws.send_text(
            SummaryUpdate(summary=state.current_summary, covered_until=0).model_dump_json()
        )
    # Action items
    if state.action_items:
        await ws.send_text(
            ActionItemsUpdate(items=state.action_items).model_dump_json()
        )


async def _ws_send(ws: WebSocket, queue: asyncio.Queue) -> None:
    """Forward events from the bus to the WebSocket client."""
    while True:
        msg = await queue.get()
        try:
            data = json.loads(msg)
            event_type = data.get("type", "unknown")
            display = data.get("display_text") or ""
            speaker = data.get("speaker", "")
            prefix = f"{speaker} " if speaker else ""
            _LLM_EVENT_TYPES = {"coach_suggestion", "reply_suggestion", "summary_update", "custom_prompt_result", "contradiction_alert"}
            log = logger.info if event_type in _LLM_EVENT_TYPES else logger.debug
            log("[web] [ws→UI] %s%s: \"%s\"", prefix, event_type, display)
        except Exception:
            logger.debug("[web] [ws→UI] (unparseable)")
        await ws.send_text(msg)


async def _ws_recv(ws: WebSocket) -> None:
    """Handle action requests from the WebSocket client."""
    while True:
        raw = await ws.receive_text()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if _context_manager is None:
            continue

        action = data.get("action")
        if action == "summary":
            asyncio.create_task(_context_manager.handle_summary_request())
        elif action == "action_items":
            asyncio.create_task(_context_manager.handle_action_items_request())
        elif action == "contradictions":
            asyncio.create_task(_context_manager.handle_contradiction_request())
        elif action == "reply":
            asyncio.create_task(_context_manager.handle_reply_request())
        elif action == "coach":
            asyncio.create_task(_context_manager.handle_coach_request())
        elif action == "custom_prompt":
            prompt = data.get("prompt", "").strip()
            if prompt:
                asyncio.create_task(_context_manager.handle_custom_prompt(prompt))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_web_app(
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    title: str = "",
    profile=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    full_transcript: bool = False,
) -> None:
    """Entry point for ``autonote realtime-web``."""
    global _api_key, _model, _meeting_title, _profile, _full_transcript
    _api_key = api_key
    _model = model
    _meeting_title = title
    _profile = profile
    _full_transcript = full_transcript

    from autonote.logger import configure_file_logging, configure_json_logging
    log_path = configure_file_logging("autonote_realtime_web")
    json_log_path = configure_json_logging("autonote_realtime_web")
    logger.info("Log file: %s", log_path.resolve())
    logger.info("Structured log: %s", json_log_path.resolve())

    import uvicorn

    logger.info("Starting web UI at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
