# Realtime Coach Mode

Coach mode turns autonote into a live negotiation/meeting coach. While you're in a call, it listens to both sides of the conversation and provides real-time tactical suggestions based on a pre-written mission brief.

## How it works

### Audio pipeline

Autonote captures two separate audio streams:

- **Microphone** → labeled `"Me"` — your voice
- **System audio** → labeled `"Them"` — the other person's voice (from the call app)

This gives deterministic speaker attribution without diarization.

### Turn aggregation

Raw transcript segments are grouped into **turns** by the `TurnAggregator`. A turn flushes (becomes complete) when any of these triggers fire:

1. **Speaker change** — the other person starts talking
2. **Silence gap** — same speaker pauses longer than `silence_threshold` (default 2.0s, configurable per profile)
3. **Max duration** — same speaker talks longer than `max_turn_duration` (default 15s, configurable)
4. **Wall-clock silence** — a background timer checks every 100ms for real-time silence

This means a single person's continuous speech can produce multiple turns if they pause mid-sentence. With `silence_threshold: 2.5`, a 3-second pause splits the speech into two turns.

### Coach triggering

When a turn completes, the `ContextManager` decides whether to invoke the coach LLM. Three rules govern this:

**1. Speaker filter** — The coach only fires on `"Them"` turns (the other person). When you're speaking, you don't need coaching — you're already talking. The coach reacts to what the other person says so you can prepare your response.

**2. Turn-count threshold** — Controlled by `coach_every_n_turns` in the profile (default: 3). Setting it to `1` means every "Them" turn triggers the coach.

**3. In-flight guard** — If a coach LLM call is already running, new triggers are suppressed. Instead, a single "pending" flag is set. When the in-flight call completes, if pending is set, one follow-up call fires with the latest context. This prevents stacking dozens of concurrent LLM calls when turns arrive faster than the model can respond.

The practical effect: the LLM's own response time (~10-15s for Opus) becomes a natural throttle. You get a coach response roughly every 10-15 seconds, each one based on the most recent context available at dispatch time. The follow-up mechanism ensures you always get one final response that captures any turns that arrived during the previous call.

### What the coach sees

Each coach invocation receives:

- **Mission brief** — your goal, role, context, prepared arguments, and coaching instructions from the YAML profile
- **Full meeting context** — running summary + action items + recent transcript (configurable window, or full transcript if `full_transcript` mode is on)
- **Recent exchange** — the last few turns (default 5, or unlimited in full-transcript mode)

The coach returns a JSON response with:
- `should_speak` — whether this is the right moment to act
- `suggestion` — what to say or do
- `argument_used` — which prepared argument from the brief is relevant
- `reasoning` — why now (or why not)
- `confidence` — high/medium/low

## Creating a coach profile

Coach profiles are YAML files. Here's a minimal example:

```yaml
name: "Job Offer Negotiation"
goal: "Secure $12,000/month with PTO matching"
role: "Felipe — Senior Engineer, sole candidate"

context: |
  The recruiter will present an offer on a 30-minute call.
  She is not the decision-maker.

arguments:
  - "This is founding engineer scope — market data puts that at $140K-$215K annually."
  - "I'm currently in a stable position. The offer needs to represent a meaningful step up."

instructions: |
  Coaching style: calm, warm, and firm.
  NEVER let the user accept on the spot — always ask for details in writing.

# Mode configuration
summary_enabled: false
action_items_enabled: false
contradictions_enabled: false
coach_every_n_turns: 1
reply_every_n_turns: 0

# Panel visibility
panels:
  summary: false
  action_items: false
  alerts: false
  coach: true
  debug: true

# Aggregator tuning
silence_threshold: 2.5
max_turn_duration: 15.0
```

### Key configuration fields

| Field | Default | Description |
|-------|---------|-------------|
| `coach_every_n_turns` | 3 | Fire coach every N "Them" turns. Set to 1 for maximum responsiveness. |
| `silence_threshold` | 2.0 | Seconds of silence before flushing a turn. Lower = more turns, more granular. |
| `max_turn_duration` | 15.0 | Force-flush a turn after this many seconds of continuous speech. |
| `summary_enabled` | true | Set false to disable background summarization (saves LLM calls). |
| `action_items_enabled` | true | Set false to disable action item scanning. |
| `contradictions_enabled` | true | Set false to disable contradiction checking. |

For a pure coaching session where you only want the coach panel, disable summary/action_items/contradictions to avoid extra LLM calls.

## Running

```bash
# Web UI (browser-based)
autonote realtime-web --profile perigon_negotiation_coach.yaml

# TUI (terminal-based)
autonote realtime --profile perigon_negotiation_coach.yaml
```

## Cost considerations

Coach mode can be expensive depending on the model and session length. The main cost drivers:

1. **Model choice** — Opus vs Sonnet vs Haiku makes 10-50x difference
2. **Number of calls** — controlled by `coach_every_n_turns` and the in-flight guard
3. **Prompt growth** — the transcript grows over time, so later calls use more input tokens

Example from a 25-minute session with Opus and `coach_every_n_turns: 1`:
- With in-flight guard + speaker filter: ~40-50 calls, ~$2-3
- Without guards (prior behavior): 267 calls, $13.27

## Analyzing LLM usage

Use the metrics script to analyze a session's JSONL log:

```bash
python scripts/parse_llm_metrics.py <session>.jsonl
```

This shows per-invocation details (time, task, model, tokens, cost, gap between calls) and aggregate summaries broken down by task type and model.
