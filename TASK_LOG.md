# Realtime Turn Grouping — Task Log

Reference: [REALTIME_TURN_GROUPING_PLAN.md](./REALTIME_TURN_GROUPING_PLAN.md)

---

## Phase 1: Timestamp-Based Turn Aggregation

- [x] Task 1.1 — Add `AggregatedTurn` model to `src/autonote/realtime/models.py` and write unit tests in `tests/unit/test_realtime_models.py`. Commit: `feat(realtime): add AggregatedTurn model for turn grouping` — 8 new tests added, 337 total passing.

  **Details:**
  - Add `AggregatedTurn` Pydantic model with fields: `speaker`, `text`, `timestamp_start`, `timestamp_end`, `segment_count`, `wall_time_start`, `wall_time_end`
  - Add `AggregatedTurn` to the `RealtimeEvent` union type
  - Tests: creation with valid fields, serialization round-trip, type discriminator is `"aggregated_turn"`, `segment_count` must be positive

- [x] Task 1.2 — Implement `TurnAggregator` class in `src/autonote/realtime/aggregator.py` (new file) and write unit tests in `tests/unit/test_turn_aggregator.py`. Commit: `feat(realtime): implement TurnAggregator with timestamp-based grouping` — 22 new tests, 359 total passing.

  **Details:**
  - Configurable `silence_threshold` (default 2.0s) and `max_turn_duration` (default 30.0s)
  - `feed(segment)`: receives `TranscriptSegment`, buffers finals, forwards partials
  - `_should_flush(incoming)`: returns True on speaker change, silence gap > threshold, or turn duration > max
  - `_flush()`: merges buffered segments into `AggregatedTurn`, puts on output queue
  - `flush_remaining()`: explicit flush for session end
  - Output queue emits `AggregatedTurn | TranscriptSegment (partials) | None (sentinel)`
  - Tests: single segment → one turn, same-speaker merge, speaker change flush, silence gap threshold, max duration flush, partials forwarded immediately, flush_remaining, empty buffer no-op, text join, timestamp_start/end, segment_count

- [x] Task 1.3 — Wire `TurnAggregator` into `src/autonote/realtime/app.py` pipeline. Commit: `feat(realtime): wire TurnAggregator into realtime pipeline` — 11 new tests, 370 total passing. Added `_bridge_segments()` task, updated `_consume_segments()` to read from `aggregator.output_queue`, added `_update_transcript_turn()` for `AggregatedTurn` TUI display, `flush_remaining()` called on quit.

  **Details:**
  - Instantiate `TurnAggregator` in `_start_pipeline()`
  - New async task reads from `transcriber.segment_queue`, feeds into `aggregator.feed()`
  - `_consume_segments()` reads from `aggregator.output_queue` instead of `transcriber.segment_queue`
  - Handle both `AggregatedTurn` and `TranscriptSegment` (partials) in consumer
  - Call `aggregator.flush_remaining()` in `action_quit`
  - Manual test: lines are grouped, partials appear, stopping flushes buffer, no data loss

- [x] Task 1.4 — Update `_append_transcript()` in `app.py` to write `AggregatedTurn` records to transcript JSONL. Commit: `feat(realtime): write aggregated turns to transcript JSONL` — 3 new tests, 373 total passing. Added `segment_count` to JSONL for `AggregatedTurn`; `TranscriptSegment` writes unchanged (backward compat).

  **Details:**
  - Each JSONL line: `speaker`, `text`, `start`, `end`, `segment_count`, `wall_time`
  - Backward compatible format readable by post-processing tools
  - Test: `AggregatedTurn.model_dump()` produces expected JSON structure

- [x] Task 1.5 — Update TUI display in `app.py` for aggregated turns with live partial line. Commit: `feat(realtime): improve TUI display with live partials and grouped turns` — 19 new tests, 392 total passing. Added `PartialLine` widget (Static, `height: 1`); extracted `_format_timestamp()` and `_speaker_style()` as static methods; `_update_partial_line()` updates live line on each partial; `_clear_partial_line()` clears it when a completed turn arrives; `_update_transcript_turn()` now clears partial before writing.

  **Details:**
  - Render `AggregatedTurn` as `[M:SS] Speaker: full text`
  - Partials: dimmed/italic "live line" that overwrites in place (use Textual `RichLog` or dedicated `Static` widget)
  - When `AggregatedTurn` arrives, clear partial line and append final turn
  - Manual test: partial text appears dimmed and updates in real time, completed turns appear as permanent colored lines, timestamps show M:SS format

---

## Phase 2: Context Manager Turn Awareness

- [x] Task 2.1 — Add `turns` list to `MeetingState` in `src/autonote/reasoning/context_manager.py`. Commit: `feat(reasoning): add turn-aware state tracking to MeetingState` — 7 new tests, 399 total passing. Added `turns: list[AggregatedTurn]`, `turns_since_last_summary`, `turns_since_last_action_scan` to `MeetingState`; added `add_turn()` method; `add_segment()` unchanged. Tests added to `test_reasoning_context_manager.py`.

  **Details:**
  - Add `turns: list[AggregatedTurn]` to `MeetingState` dataclass
  - Add `turns_since_last_summary: int` and `turns_since_last_action_scan: int` counters
  - Add `add_turn(turn: AggregatedTurn)` method that appends and increments counters
  - Keep `add_segment()` working for backward compatibility
  - Tests (new `tests/unit/test_context_manager.py`): `add_turn()` increments counters, turns list grows, speakers set updated

- [x] Task 2.2 — Switch reasoning triggers from segment-count to turn-count in `context_manager.py`. Commit: `feat(reasoning): switch reasoning triggers to turn-based thresholds` — 12 new tests, 411 total passing. Added `on_new_turn()` method with `SUMMARY_EVERY_N_TURNS=5` and `ACTION_SCAN_EVERY_N_TURNS=3`; `on_new_segment()` unchanged for backward compat; `app.py` now calls `on_new_turn(item)` directly instead of synthetic segment. Updated 2 existing tests in `test_realtime_app.py` that expected synthetic `TranscriptSegment`.

  **Details:**
  - Change `on_new_segment()` to `on_new_turn(turn: AggregatedTurn)`
  - Summary worker triggers every 5 turns (was every 10 segments)
  - Action item worker triggers every 3 turns (was every 5 segments)
  - Contradiction check remains time-based (every 120s)
  - Tests: summary triggers after 5 turns, action items after 3 turns, contradiction still time-based

- [ ] Task 2.3 — Add `get_turn_transcript()` to `context_manager.py` and update reasoning workers to use it. Commit: `feat(reasoning): add turn-based transcript formatting for LLM context`

  **Details:**
  - `get_turn_transcript(last_n: int | None = None) -> str` formats turns as `Speaker: text` blocks
  - Update reasoning workers to call `get_turn_transcript()` instead of `get_transcript_text()` where applicable
  - Keep `get_transcript_text()` for backward compat
  - Tests: expected format output, `last_n` limits to most recent N turns

---

## Phase 3: Local VAD Integration

- [ ] Task 3.1 — Move `SileroVAD` from `meeting-copilot/backend/audio/vad.py` into `src/autonote/realtime/vad.py`. Commit: `refactor(realtime): move SileroVAD into autonote.realtime package`

  **Details:**
  - Add `SpeechStateEvent` model to `models.py`: `speaker`, `event_type` (speech_start | speech_end), `timestamp`, `silence_duration`
  - Tests (new `tests/unit/test_vad.py`): model loads, `is_speech()` returns True/False correctly, `SpeechStateEvent` serialization

- [ ] Task 3.2 — Create `VADMonitor` async worker in `src/autonote/realtime/vad_monitor.py`. Commit: `feat(realtime): add VADMonitor for real-time speech state tracking`

  **Details:**
  - Reads raw PCM chunks from a queue (tee'd from recorder)
  - Runs Silero VAD on each chunk
  - Tracks speech state transitions (speaking→silent, silent→speaking)
  - Emits `SpeechStateEvent` to output queue on transitions
  - Configurable silence threshold (default 1.5s of continuous silence = speech_end)
  - Tests (new `tests/unit/test_vad_monitor.py`): speech→silence emits after threshold, brief pause does not emit, silence→speech emits speech_start, continuous speech no spurious events

- [ ] Task 3.3 — Tee recorder output to both transcriber and VAD queues in `src/autonote/realtime/recorder.py`. Commit: `feat(realtime): tee audio chunks to both transcriber and VAD queues`

  **Details:**
  - Add second queue per stream (`mic_vad_queue`, `monitor_vad_queue`)
  - Reader loop puts each chunk into both transcriber and VAD queue
  - Expose VAD queues via properties
  - Tests (new `tests/unit/test_recorder_tee.py`): chunks appear in both queues, sentinel (None) sent to both on stop

- [ ] Task 3.4 — Add `feed_vad_event()` to `TurnAggregator` in `aggregator.py`. Commit: `feat(realtime): integrate VAD events into TurnAggregator for faster flush`

  **Details:**
  - `feed_vad_event(event: SpeechStateEvent)` method
  - On `speech_end` with silence > threshold, flush buffer for that speaker immediately
  - On `speech_start` from other speaker, flush current speaker's buffer
  - Tests: VAD speech_end flushes without waiting for next segment, cross-speaker speech_start triggers flush, VAD events with no buffered segments are no-ops, interaction between VAD events and regular segment feed

- [ ] Task 3.5 — Wire VAD pipeline into `app.py`. Commit: `feat(realtime): wire VAD-based silence detection into realtime pipeline`

  **Details:**
  - Start `VADMonitor` tasks for mic and monitor streams in `_start_pipeline()`
  - Create async task to feed VAD events into aggregator
  - Shut down VAD monitors in `action_quit()`
  - Manual test: faster turn boundaries, no latency increase, clean shutdown

---

## Phase 4: Cross-Stream Overlap Detection

- [ ] Task 4.1 — Add overlap detection to `VADMonitor` in `vad_monitor.py`. Commit: `feat(realtime): detect speaker overlap via dual-stream VAD`

  **Details:**
  - Track speech state for both streams simultaneously
  - Detect when both mic and monitor are in `speech` state (crosstalk/interruption)
  - Emit `OverlapEvent` model with: `timestamp_start`, `timestamp_end`, `duration`
  - Tests: simultaneous speech emits `OverlapEvent`, non-overlapping speech does not, overlap duration calculated correctly

- [ ] Task 4.2 — Surface overlaps in TUI and reasoning in `app.py` and `context_manager.py`. Commit: `feat(realtime): surface speaker overlaps in TUI and reasoning context`

  **Details:**
  - Show overlap indicator in TUI (e.g., `[crosstalk]` annotation between turns)
  - Pass overlap metadata to reasoning workers (e.g., "heated exchange with overlapping speech")
  - Manual test: overlaps displayed and mentioned in summaries when relevant

---

## Phase 5: Per-Turn Reasoning Improvements

- [ ] Task 5.1 — Change reply suggestion trigger in `src/autonote/reasoning/workers/reply.py` to fire on complete turns only. Commit: `feat(reasoning): trigger reply suggestions on complete turns only`

  **Details:**
  - Trigger only when "Them" completes a full `AggregatedTurn` (not on partial fragments)
  - Use complete turn text as prompt for reply generation
  - Detect questions in turn text for auto-triggering
  - Tests: reply triggers after `AggregatedTurn` from "Them", partials do not trigger reply

- [ ] Task 5.2 — Create conversation pattern detection worker in `src/autonote/reasoning/workers/patterns.py`. Commit: `feat(reasoning): detect conversation patterns from turn sequences`

  **Details:**
  - Analyze turn sequence to detect patterns: Q&A, monologue, debate, brainstorm
  - Expose pattern as metadata in `MeetingState`
  - Use detected pattern to adjust summary style (e.g., Q&A → bullet list of Q&As)
  - Tests (new `tests/unit/test_patterns.py`): alternating short turns → Q&A, single long-turn speaker → monologue, rapid equal-length alternation → debate
