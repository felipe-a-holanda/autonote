# Build Context — Meeting Copilot
> Atualizado automaticamente pelo agente após cada iteração.

## Estado Atual
- **Fase**: 2 — Context Manager Turn Awareness
- **Última tarefa**: Task 2.3 — Add turn-based transcript formatting for LLM context
- **Testes passando**: 422

## Decisões Técnicas (Task 2.3)
- `get_turn_transcript()` format is `Speaker: text` (simpler than segment's `[Speaker @ Xs]: text`).
- `get_full_context()` prefers `get_turn_transcript(last_n=30)` when `self.turns` is non-empty.
- `_run_summary()` uses `get_turn_transcript(last_n=SUMMARY_EVERY_N_TURNS)` when turns exist; `covered_until` taken from `turns[-1].timestamp_end` in that case.
- `_run_contradictions()` and `_run_action_items()` similarly prefer turn transcript.
- All three `_run_*` methods fall back to segment-based transcript if no turns exist.

## Decisões Técnicas (Task 2.2)
- `on_new_segment()` kept as-is for backward compat; `on_new_turn()` is the new primary trigger path.
- `app.py` now calls `on_new_turn(item)` directly — no more synthetic `TranscriptSegment` wrapper.
- Two `test_realtime_app.py` tests updated: they now assert `AggregatedTurn` is passed to `on_new_turn`, not a synthetic segment.
- `SUMMARY_EVERY_N_TURNS=5` and `ACTION_SCAN_EVERY_N_TURNS=3` are overridable via constructor kwargs `summary_every_n_turns` / `action_scan_every_n_turns`.

## Decisões Técnicas (Task 2.1)
- `turns` and `turns_since_*` counters are independent from `segments` and `segments_since_*` — `add_turn()` does not touch segment counters and vice versa.
- `add_turn()` does not update `recent_window` — that window is segment-based; turn-based window added in a later task if needed.
- Tests added to existing `test_reasoning_context_manager.py` (not a new file) for cohesion.

## Decisões Técnicas (Task 1.5)
- `PartialLine` is a `Static` subclass with `height: 1`; sits below `TranscriptLog` inside a `Vertical(id="transcript-container")`.
- `_format_timestamp()` and `_speaker_style()` extracted as `@staticmethod` — pure, easily testable.
- `_update_partial_line()` and `_clear_partial_line()` swallow exceptions (no DOM outside running app).
- `_update_transcript()` for partials now calls `_update_partial_line()` instead of returning early.
- `_update_transcript_turn()` calls `_clear_partial_line()` before writing to RichLog.

## Decisões Técnicas
- `wall_time_start` / `wall_time_end` on `AggregatedTurn` are `Optional[datetime]` (not float), since they represent wall clock time rather than session-relative audio timestamps.
- `segment_count` validated via `@field_validator` to enforce >= 1.
- `AggregatedTurn` inserted before `SummaryUpdate` in the `RealtimeEvent` union.
- `TurnAggregator.feed()` is synchronous; uses `asyncio.Queue.put_nowait()` so it can be called from an async task without await.
- `_should_flush()` uses strictly-greater-than for silence gap (gap > threshold, not >=).
- `flush_remaining()` is a no-op when buffer is empty — no sentinel is put on the queue; the app layer is responsible for signalling consumers separately.
- Pipeline in app.py: transcriber→bridge_segments→aggregator→consume_segments. Bridge puts None sentinel on aggregator queue after flush_remaining().
- `_consume_segments()` reads from `aggregator.output_queue`; for `AggregatedTurn`, creates a synthetic `TranscriptSegment` to pass to context_manager (until Task 2.x updates it).
- `_append_transcript()` accepts `TranscriptSegment | AggregatedTurn`; for `AggregatedTurn` it also writes `segment_count` to JSONL; `TranscriptSegment` records omit it (backward compat).
- Tests using `asyncio.Queue` must use `asyncio.get_event_loop().run_until_complete()`, NOT `asyncio.run()` — the latter destroys the event loop and breaks other tests that use `asyncio.get_event_loop()`.

## Problemas Conhecidos / Armadilhas
- Full test suite runs with `uv run pytest` (not `python -m pytest`; .venv python lacks pytest).
- Three pre-existing RuntimeWarning about unawaited coroutines in test mocks — not new, not blocking.
- `asyncio.run()` in tests breaks subsequent tests that call `asyncio.get_event_loop()` — use `run_until_complete()` instead.

## Arquivos Críticos
- `src/autonote/realtime/models.py` — added `AggregatedTurn`, updated `RealtimeEvent` union
- `src/autonote/realtime/aggregator.py` — TurnAggregator implementation
- `src/autonote/realtime/app.py` — full pipeline with PartialLine widget, `_format_timestamp`, `_speaker_style`, `_update_partial_line`, `_clear_partial_line`
- `tests/unit/test_realtime_models.py` — 8 tests for `AggregatedTurn`
- `tests/unit/test_turn_aggregator.py` — 22 tests for `TurnAggregator`
- `tests/unit/test_realtime_app.py` — 57 tests (19 new for Task 1.5 TUI display)
