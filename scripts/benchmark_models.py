#!/usr/bin/env python3
"""
benchmark_models.py — Testa múltiplos modelos LLM na sumarização de uma mesma reunião.

Uso:
    python scripts/benchmark_models.py <transcription_file> [--models m1 m2 m3] [--parallel]

Exemplos:
    # Com modelos padrão
    python scripts/benchmark_models.py recordings/20260327/meeting_20260327_113024/meeting_20260327_113024_formatted.md

    # Com modelos específicos (suporta presets: cheap, smart, fast, local)
    python scripts/benchmark_models.py transcript.md --models cheap smart openrouter/google/gemini-flash-1.5

    # Rodar em paralelo (mais rápido, mais caro)
    python scripts/benchmark_models.py transcript.md --parallel

Saída:
    Arquivos no mesmo diretório da transcrição, com sufixo _summary_<slug_modelo>.md
    Tabela comparativa impressa no terminal ao final.
"""

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: garante que o pacote autonote do workspace está no path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from autonote.audio.summarize import load_transcription, summarize_meeting, save_summary
from autonote.llm import resolve_model
from autonote.logger import log_info, log_error

# ---------------------------------------------------------------------------
# Modelos padrão do benchmark
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    "cheap",                                        # preset → deepseek/deepseek-chat
    "smart",                                        # preset → anthropic/claude-sonnet-4-6
    "openrouter/google/gemini-flash-1.5",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_slug(model: str) -> str:
    """Converte nome de modelo em slug seguro para nome de arquivo."""
    resolved = resolve_model(model)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", resolved)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:60]


def _cost_from_log(recording_dir: Path, base_stem: str, slug: str) -> tuple[float, float | None]:
    """Lê o custo do _llm_costs.json filtrando pelas entradas com o slug do modelo."""
    cost_file = recording_dir / f"{base_stem}_llm_costs.json"
    if not cost_file.exists():
        return 0.0, None
    import json
    try:
        entries = json.loads(cost_file.read_text(encoding="utf-8"))
        resolved_model = resolve_model(slug)  # slug here is actually the original model string
        total_usd = sum(
            e.get("cost_usd", 0.0)
            for e in entries
            if e.get("model", "") == resolved_model
        )
        brl_values = [
            e.get("cost_brl")
            for e in entries
            if e.get("model", "") == resolved_model and e.get("cost_brl") is not None
        ]
        total_brl = sum(brl_values) if brl_values else None
        return total_usd, total_brl
    except Exception:
        return 0.0, None


def _run_one(transcription_file: str, model: str, transcription: str) -> dict:
    """Executa summarização para um modelo e retorna métricas."""
    resolved = resolve_model(model)
    slug = _model_slug(model)
    trans_path = Path(transcription_file)
    base_stem = re.sub(r"(_formatted|_summary|_extracted_metadata)$", "", trans_path.stem)
    output_file = trans_path.parent / f"{base_stem}_summary_{slug}.md"

    log_info(f"[benchmark] Starting model: {resolved}")
    t0 = time.monotonic()
    try:
        result = summarize_meeting(
            transcription,
            model=model,
            ollama_url=None,  # usa OLLAMA_URL do config ou default
            include_action_items=True,
            source_file=transcription_file,
        )
        elapsed = time.monotonic() - t0
        save_summary(result, str(output_file), format="md")
        log_info(f"[benchmark] Done: {resolved} ({elapsed:.1f}s) → {output_file.name}")

        cost_usd, cost_brl = _cost_from_log(trans_path.parent, base_stem, model)
        return {
            "model": model,
            "resolved": resolved,
            "slug": slug,
            "output_file": str(output_file),
            "elapsed_s": round(elapsed, 1),
            "cost_usd": cost_usd,
            "cost_brl": cost_brl,
            "error": None,
        }
    except Exception as e:
        elapsed = time.monotonic() - t0
        log_error(f"[benchmark] Failed for {resolved}: {e}")
        return {
            "model": model,
            "resolved": resolved,
            "slug": slug,
            "output_file": None,
            "elapsed_s": round(elapsed, 1),
            "cost_usd": 0.0,
            "cost_brl": None,
            "error": str(e),
        }


def _print_table(results: list[dict]) -> None:
    """Imprime tabela comparativa no terminal."""
    col_model = max(len(r["resolved"]) for r in results)
    col_model = max(col_model, 8)

    header = f"{'Model':<{col_model}}  {'Time(s)':>8}  {'USD':>10}  {'BRL':>10}  {'File'}"
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in sorted(results, key=lambda x: x["cost_usd"]):
        cost_usd = f"${r['cost_usd']:.6f}" if r["cost_usd"] else "?"
        cost_brl = f"R${r['cost_brl']:.4f}" if r["cost_brl"] else "?"
        fname = Path(r["output_file"]).name if r["output_file"] else f"ERROR: {r['error']}"
        status = "✗ " if r["error"] else ""
        print(f"{status}{r['resolved']:<{col_model}}  {r['elapsed_s']:>8.1f}  {cost_usd:>10}  {cost_brl:>10}  {fname}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark múltiplos modelos LLM na sumarização de uma reunião.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "transcription_file",
        help="Arquivo de transcrição (.txt, .md, ou .json) ou transcript formatado (_formatted.md).",
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help=(
            "Modelos a testar. Aceita presets (cheap, smart, fast, local) ou strings completas "
            "(ex: openrouter/google/gemini-flash-1.5). Padrão: %(default)s"
        ),
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Executa todos os modelos em paralelo (mais rápido, mas pode estourar rate limits).",
    )
    args = parser.parse_args()

    trans_path = Path(args.transcription_file)
    if not trans_path.exists():
        log_error(f"Arquivo não encontrado: {trans_path}")
        sys.exit(1)

    log_info(f"Carregando transcrição: {trans_path}")
    transcription = load_transcription(str(trans_path))
    if not transcription.strip():
        log_error("Transcrição está vazia.")
        sys.exit(1)

    models = args.models
    log_info(f"Modelos a testar: {models}")
    log_info(f"Modo: {'paralelo' if args.parallel else 'sequencial'}")

    results = []
    if args.parallel:
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {
                executor.submit(_run_one, str(trans_path), m, transcription): m
                for m in models
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for model in models:
            result = _run_one(str(trans_path), model, transcription)
            results.append(result)

    _print_table(results)

    # Salva resultado em JSON junto ao arquivo de transcrição
    import json
    base_stem = re.sub(r"(_formatted|_summary|_extracted_metadata)$", "", trans_path.stem)
    report_path = trans_path.parent / f"{base_stem}_benchmark_report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log_info(f"Relatório salvo: {report_path}")


if __name__ == "__main__":
    main()
