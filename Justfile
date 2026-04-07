# Autonote — common workflow commands
# Usage: just <recipe> [args]
# Requires: just (https://github.com/casey/just)

# Show available recipes
default:
    @just --list

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

# Record and run the full pipeline (optionally pass: title="My Meeting" duration=3600)
[no-exit-message]
record title="" duration="":
    autonote full \
        {{ if title != "" { '--title "' + title + '"' } else { "" } }} \
        {{ if duration != "" { '--duration ' + duration } else { "" } }}

# Record with diarization (speaker separation)
[no-exit-message]
record-diarize title="" duration="" speakers="":
    autonote full --diarize \
        {{ if title != "" { '--title "' + title + '"' } else { "" } }} \
        {{ if duration != "" { '--duration ' + duration } else { "" } }} \
        {{ if speakers != "" { '--speakers ' + speakers } else { "" } }}

# Process the most recently recorded meeting
process-last:
    autonote process-last

# ---------------------------------------------------------------------------
# Reprocessing
# ---------------------------------------------------------------------------

# Re-run summarize + obsidian sync on a specific file
reprocess file:
    autonote reprocess "{{file}}" --summarize --obsidian

# Re-run reformat + summarize + obsidian on a specific file
reprocess-full file:
    autonote reprocess "{{file}}" --reformat --summarize --obsidian

# Reprocess all meetings since a date (YYYY-MM-DD)
reprocess-since date:
    autonote reprocess --since {{date}} --reformat --summarize --obsidian

# ---------------------------------------------------------------------------
# Cost reporting
# ---------------------------------------------------------------------------

COST_LOG := env_var_or_default("LLM_COST_LOG", env_var("HOME") + "/.autonote_costs.jsonl")

# Total USD spent across all calls
cost-total:
    @jq -rs '[.[].cost_usd] | add | "Total: $\(. * 100000 | round / 100000)"' "{{COST_LOG}}"

# Cost breakdown by pipeline stage
cost-by-stage:
    @jq -rs 'group_by(.stage) | map({stage: .[0].stage, cost_usd: (map(.cost_usd) | add), calls: length}) | sort_by(.cost_usd) | reverse | .[] | "\(.stage // "unknown")\t$\(.cost_usd | . * 1000000 | round / 1000000)\t(\(.calls) calls)"' "{{COST_LOG}}" | column -t

# Cost breakdown by model
cost-by-model:
    @jq -rs 'group_by(.model) | map({model: .[0].model, cost_usd: (map(.cost_usd) | add), calls: length}) | sort_by(.cost_usd) | reverse | .[] | "\(.model)\t$\(.cost_usd | . * 1000000 | round / 1000000)\t(\(.calls) calls)"' "{{COST_LOG}}" | column -t

# Top 10 most expensive recordings
cost-by-recording:
    @jq -rs 'group_by(.recording_dir) | map({dir: .[0].recording_dir, cost_usd: (map(.cost_usd) | add)}) | sort_by(.cost_usd) | reverse | .[0:10] | .[] | "$\(.cost_usd | . * 1000000 | round / 1000000)\t\(.dir // "unknown")"' "{{COST_LOG}}" | column -t

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

# Run all tests
test:
    pytest -v

# Run tests with coverage report
test-cov:
    pytest --cov=autonote --cov-report=term-missing --cov-report=html

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# List all recordings
list:
    autonote list

# Remove all WAV/MP3 files (with confirmation)
clean:
    autonote clean
    
    
perigon:
    autonote realtime-web --profile perigon_negotiation_coach.yaml --full-transcript --model smart
    
  
