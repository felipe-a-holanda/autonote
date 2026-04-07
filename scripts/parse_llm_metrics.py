#!/usr/bin/env python3
"""Parse autonote realtime JSONL logs and print LLM invocation metrics.

Usage:
    python scripts/parse_llm_metrics.py <file.jsonl> [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def load_events(path: Path) -> tuple[list[dict], list[dict]]:
    """Stream-parse JSONL and return (requests, usages)."""
    requests: list[dict] = []
    usages: list[dict] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = entry.get("event") or entry.get("structured", {}).get("event")
            # Support both flat structured entries and wrapped ones
            if event == "llm_request":
                requests.append({
                    "ts": entry.get("ts"),
                    "task": entry.get("task"),
                    "model": entry.get("model"),
                    "prompt_len": len(entry.get("prompt", "")) if "prompt" in entry else None,
                })
            elif event == "llm_usage":
                usages.append({
                    "ts": entry.get("ts"),
                    "stage": entry.get("stage"),
                    "model": entry.get("model"),
                    "tokens_in": entry.get("tokens_in"),
                    "tokens_out": entry.get("tokens_out"),
                    "tokens_total": entry.get("tokens_total"),
                    "cost_usd": entry.get("cost_usd"),
                    "duration_s": entry.get("duration_s"),
                })

    return requests, usages


def pair_events(requests: list[dict], usages: list[dict]) -> list[dict]:
    """Join requests with their corresponding usage events by task/stage name."""
    # Index usages by stage name, preserving order
    usage_queues: dict[str, list[dict]] = defaultdict(list)
    for u in usages:
        stage = u.get("stage") or ""
        # stage is stored as "realtime_summary" → strip prefix
        key = stage.replace("realtime_", "")
        usage_queues[key].append(u)

    # Usage queue pointers
    usage_ptrs: dict[str, int] = defaultdict(int)

    paired = []
    for req in requests:
        task = req.get("task") or ""
        queue = usage_queues.get(task, [])
        ptr = usage_ptrs[task]
        usage = queue[ptr] if ptr < len(queue) else None
        usage_ptrs[task] += 1

        entry: dict = {
            "ts": req["ts"],
            "task": task,
            "model": req.get("model"),
            "prompt_len": req.get("prompt_len"),
        }
        if usage:
            entry.update({
                "model": usage.get("model") or entry.get("model"),
                "tokens_in": usage.get("tokens_in"),
                "tokens_out": usage.get("tokens_out"),
                "tokens_total": usage.get("tokens_total"),
                "cost_usd": usage.get("cost_usd"),
                "duration_s": usage.get("duration_s"),
            })
        paired.append(entry)

    return paired


def compute_gaps(paired: list[dict]) -> list[Optional[float]]:
    gaps: list[Optional[float]] = [None]
    for i in range(1, len(paired)):
        try:
            t_prev = parse_ts(paired[i - 1]["ts"])
            t_cur = parse_ts(paired[i]["ts"])
            gaps.append((t_cur - t_prev).total_seconds())
        except Exception:
            gaps.append(None)
    return gaps


def fmt_num(n: Optional[int | float], decimals: int = 0) -> str:
    if n is None:
        return "—"
    if decimals:
        return f"{n:,.{decimals}f}"
    return f"{int(n):,}"


def _short_model(model: Optional[str]) -> str:
    """Strip provider prefix for compact display (e.g. 'deepseek/deepseek-chat' → 'deepseek-chat')."""
    if not model:
        return "—"
    return model.split("/")[-1]


def print_table(paired: list[dict], gaps: list[Optional[float]], filename: str) -> None:
    print(f"\nLLM Invocation Metrics — {filename}")
    print("=" * 90)
    print(f"{'#':>3}  {'Time':>8}  {'Task':<16}  {'Model':<20}  {'Prompt':>8}  {'Tok-in':>7}  {'Tok-out':>7}  {'Dur(s)':>7}  {'Gap(s)':>7}")
    print("-" * 90)

    for i, (row, gap) in enumerate(zip(paired, gaps), start=1):
        ts_str = "—"
        if row.get("ts"):
            try:
                ts_str = parse_ts(row["ts"]).strftime("%H:%M:%S")
            except Exception:
                pass

        gap_str = f"{gap:.1f}" if gap is not None else "—"
        dur_str = f"{row['duration_s']:.1f}" if row.get("duration_s") is not None else "—"

        print(
            f"{i:>3}  {ts_str:>8}  {row.get('task', '?'):<16}  "
            f"{_short_model(row.get('model')):<20}  "
            f"{fmt_num(row.get('prompt_len')):>8}  "
            f"{fmt_num(row.get('tokens_in')):>7}  "
            f"{fmt_num(row.get('tokens_out')):>7}  "
            f"{dur_str:>7}  "
            f"{gap_str:>7}"
        )

    print()
    _print_summary(paired, gaps)


def _print_summary(paired: list[dict], gaps: list[Optional[float]]) -> None:
    n = len(paired)
    if n == 0:
        print("No LLM invocations found.")
        return

    prompt_lens = [r["prompt_len"] for r in paired if r.get("prompt_len") is not None]
    valid_gaps = [g for g in gaps if g is not None]
    total_tok_in = sum(r["tokens_in"] for r in paired if r.get("tokens_in") is not None)
    total_tok_out = sum(r["tokens_out"] for r in paired if r.get("tokens_out") is not None)
    total_cost = sum(r["cost_usd"] for r in paired if r.get("cost_usd") is not None)

    task_counts: dict[str, int] = defaultdict(int)
    for r in paired:
        task_counts[r.get("task") or "unknown"] += 1

    # Per-model aggregates: {model: {calls, tokens_in, tokens_out, cost}}
    model_stats: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
    for r in paired:
        m = _short_model(r.get("model"))
        model_stats[m]["calls"] += 1
        model_stats[m]["tokens_in"] += r.get("tokens_in") or 0
        model_stats[m]["tokens_out"] += r.get("tokens_out") or 0
        model_stats[m]["cost_usd"] += r.get("cost_usd") or 0.0

    print("Summary")
    print("-" * 40)
    print(f"Total invocations : {n}")
    if prompt_lens:
        print(f"Avg prompt size   : {sum(prompt_lens) // len(prompt_lens):,} chars  (max: {max(prompt_lens):,})")
    if valid_gaps:
        avg_gap = sum(valid_gaps) / len(valid_gaps)
        print(f"Avg gap           : {avg_gap:.1f}s  (min: {min(valid_gaps):.1f}s  max: {max(valid_gaps):.1f}s)")
    if total_tok_in or total_tok_out:
        print(f"Total tokens      : {total_tok_in + total_tok_out:,}  (in: {total_tok_in:,} / out: {total_tok_out:,})")
    if total_cost:
        print(f"Total cost        : ${total_cost:.4f}")
    print()
    print("Calls by task:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task:<20} {count}")
    print()
    print("Calls by model:")
    print(f"  {'Model':<28}  {'Calls':>5}  {'Tok-in':>8}  {'Tok-out':>8}  {'Cost':>10}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}")
    for model, s in sorted(model_stats.items(), key=lambda x: -x[1]["cost_usd"]):
        cost_str = f"${s['cost_usd']:.4f}" if s["cost_usd"] else "—"
        print(f"  {model:<28}  {s['calls']:>5}  {s['tokens_in']:>8,}  {s['tokens_out']:>8,}  {cost_str:>10}")
    print()


def output_json(paired: list[dict], gaps: list[Optional[float]]) -> None:
    rows = []
    for row, gap in zip(paired, gaps):
        rows.append({**row, "gap_s": gap})

    valid_gaps = [g for g in gaps if g is not None]
    prompt_lens = [r["prompt_len"] for r in paired if r.get("prompt_len") is not None]
    total_tok_in = sum(r["tokens_in"] for r in paired if r.get("tokens_in") is not None)
    total_tok_out = sum(r["tokens_out"] for r in paired if r.get("tokens_out") is not None)
    total_cost = sum(r["cost_usd"] for r in paired if r.get("cost_usd") is not None)

    model_stats: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
    for r in paired:
        m = _short_model(r.get("model"))
        model_stats[m]["calls"] += 1
        model_stats[m]["tokens_in"] += r.get("tokens_in") or 0
        model_stats[m]["tokens_out"] += r.get("tokens_out") or 0
        model_stats[m]["cost_usd"] = round(model_stats[m]["cost_usd"] + (r.get("cost_usd") or 0.0), 8)

    output = {
        "total_invocations": len(paired),
        "avg_prompt_chars": (sum(prompt_lens) // len(prompt_lens)) if prompt_lens else None,
        "avg_gap_s": (sum(valid_gaps) / len(valid_gaps)) if valid_gaps else None,
        "min_gap_s": min(valid_gaps) if valid_gaps else None,
        "max_gap_s": max(valid_gaps) if valid_gaps else None,
        "total_tokens_in": total_tok_in,
        "total_tokens_out": total_tok_out,
        "total_cost_usd": round(total_cost, 6),
        "by_model": dict(model_stats),
        "invocations": rows,
    }
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse autonote JSONL logs for LLM metrics.")
    parser.add_argument("file", help="Path to .jsonl log file")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    requests, usages = load_events(path)
    if not requests:
        print("No llm_request events found in log.", file=sys.stderr)
        sys.exit(0)

    paired = pair_events(requests, usages)
    gaps = compute_gaps(paired)

    if args.json:
        output_json(paired, gaps)
    else:
        print_table(paired, gaps, path.name)


if __name__ == "__main__":
    main()
