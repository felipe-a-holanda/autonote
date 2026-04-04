# Build Context — Meeting Copilot
> Atualizado automaticamente pelo agente após cada iteração.

## Estado Atual
- **Fase**: 1 — Timestamp-Based Turn Aggregation
- **Última tarefa**: Task 1.2 — TurnAggregator implemented in aggregator.py
- **Testes passando**: 359

## Decisões Técnicas
- `wall_time_start` / `wall_time_end` on `AggregatedTurn` are `Optional[datetime]` (not float), since they represent wall clock time rather than session-relative audio timestamps.
- `segment_count` validated via `@field_validator` to enforce >= 1.
- `AggregatedTurn` inserted before `SummaryUpdate` in the `RealtimeEvent` union.
- `TurnAggregator.feed()` is synchronous; uses `asyncio.Queue.put_nowait()` so it can be called from an async task without await.
- `_should_flush()` uses strictly-greater-than for silence gap (gap > threshold, not >=).
- `flush_remaining()` is a no-op when buffer is empty — no sentinel is put on the queue; the app layer is responsible for signalling consumers separately.

## Problemas Conhecidos / Armadilhas
- Full test suite runs with `uv run pytest` (not `python -m pytest`; .venv python lacks pytest).
- Three pre-existing RuntimeWarning about unawaited coroutines in test mocks — not new, not blocking.

## Arquivos Críticos
- `src/autonote/realtime/models.py` — added `AggregatedTurn`, updated `RealtimeEvent` union
- `src/autonote/realtime/aggregator.py` — TurnAggregator implementation
- `tests/unit/test_realtime_models.py` — 8 tests for `AggregatedTurn`
- `tests/unit/test_turn_aggregator.py` — 22 tests for `TurnAggregator`
