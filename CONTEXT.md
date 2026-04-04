# Build Context — Meeting Copilot
> Atualizado automaticamente pelo agente após cada iteração.

## Estado Atual
- **Fase**: 1 — Timestamp-Based Turn Aggregation
- **Última tarefa**: Task 1.4 — `_append_transcript()` writes `segment_count` for `AggregatedTurn`
- **Testes passando**: 373

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
- `src/autonote/realtime/app.py` — pipeline wired with aggregator; `_bridge_segments`, `_consume_segments`, `_update_transcript_turn`
- `tests/unit/test_realtime_models.py` — 8 tests for `AggregatedTurn`
- `tests/unit/test_turn_aggregator.py` — 22 tests for `TurnAggregator`
- `tests/unit/test_realtime_app.py` — 35 tests (11 new for Task 1.3 wiring)
