# Build Context — Meeting Copilot
> Atualizado automaticamente pelo agente após cada iteração.

## Estado Atual
- **Fase**: 1 — Timestamp-Based Turn Aggregation
- **Última tarefa**: Task 1.1 — AggregatedTurn model added to models.py
- **Testes passando**: 337

## Decisões Técnicas
- `wall_time_start` / `wall_time_end` on `AggregatedTurn` are `Optional[datetime]` (not float), since they represent wall clock time rather than session-relative audio timestamps.
- `segment_count` validated via `@field_validator` to enforce >= 1.
- `AggregatedTurn` inserted before `SummaryUpdate` in the `RealtimeEvent` union.

## Problemas Conhecidos / Armadilhas
- Full test suite runs with `uv run pytest` (not `python -m pytest`; .venv python lacks pytest).
- Two pre-existing RuntimeWarning about unawaited coroutines in test mocks — not new, not blocking.

## Arquivos Críticos
- `src/autonote/realtime/models.py` — added `AggregatedTurn`, updated `RealtimeEvent` union
- `tests/unit/test_realtime_models.py` — 8 new tests for `AggregatedTurn`
