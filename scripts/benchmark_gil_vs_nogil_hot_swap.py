#!/usr/bin/env python3
"""Compare hot-swap benchmark results with PYTHON_GIL=1 vs PYTHON_GIL=0."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_ROOT / ".venv314t" / "bin" / "python"
METRICS = (
    "ttfb_seconds",
    "first_ar_iteration_wall_seconds",
    "prefill_seconds",
    "swap_to_decode_seconds",
    "decode_load_wait_seconds",
    "first_ar_iteration_seconds",
    "generation_seconds",
    "generation_seconds_excluding_decode_wait",
    "wall_seconds",
    "rtf",
    "rtf_excluding_decode_wait",
)


def _parse_args() -> argparse.Namespace:
    timestamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Run voxcpmane2-benchmark-overlap twice in hot-swap mode: once "
            "with PYTHON_GIL=1 and once with PYTHON_GIL=0."
        )
    )
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON),
        help=f"Free-threaded Python executable. Default: {DEFAULT_PYTHON}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmarks" / f"gil-vs-nogil-hot-swap-{timestamp}"),
        help="Directory for raw logs and comparison.json.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--variant", choices=("baseline", "async", "both"), default="async")
    parser.add_argument("--lm-prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running the benchmark.",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments after '--' are forwarded to voxcpmane2-benchmark-overlap, "
            "for example: -- --model-dir /path/to/models --prompt-wav-path ..."
        ),
    )
    return parser.parse_args()


def _strip_separator(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def _has_option(args: list[str], option: str) -> bool:
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in args)


def _benchmark_command(args: argparse.Namespace, benchmark_args: list[str]) -> list[str]:
    cmd = [args.python, "-m", "voxcpmane.benchmark"]
    defaults = {
        "--variant": args.variant,
        "--lm-mode": "hot-swap",
        "--runs": str(args.runs),
        "--warmup-runs": str(args.warmup_runs),
        "--lm-prefill-chunk-size": str(args.lm_prefill_chunk_size),
    }
    for option, value in defaults.items():
        if not _has_option(benchmark_args, option):
            cmd.extend([option, value])
    cmd.extend(benchmark_args)
    return cmd


def _extract_summary(log_text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(log_text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(log_text[index:])
        except json.JSONDecodeError:
            continue
        summary = obj.get("summary") if isinstance(obj, dict) else None
        if (
            isinstance(summary, dict)
            and isinstance(summary.get("python"), dict)
            and isinstance(summary.get("variants"), list)
        ):
            return summary
    raise RuntimeError("benchmark output did not contain a final summary JSON object")


def _run_mode(
    *,
    label: str,
    gil_value: str,
    command: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    log_path = output_dir / f"{label}.log"
    env = os.environ.copy()
    env["PYTHON_GIL"] = gil_value
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"\n== {label}: PYTHON_GIL={gil_value} ==")
    print(" ".join(command))
    proc = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in proc.stdout:
            chunks.append(line)
            log_file.write(line)
            print(f"[{label}] {line}", end="")
    return_code = proc.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    summary = _extract_summary("".join(chunks))
    expected = gil_value == "1"
    actual = summary.get("python", {}).get("gil_enabled")
    if actual is not None and bool(actual) != expected:
        print(
            f"[{label}] warning: expected gil_enabled={expected}, "
            f"benchmark reported {actual}",
            file=sys.stderr,
        )
    (output_dir / f"{label}.summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def _variant_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(variant["name"]): variant
        for variant in summary.get("variants", [])
        if isinstance(variant, dict) and "name" in variant
    }


def _mean(variant: dict[str, Any], metric: str) -> float | None:
    value = variant.get("summary", {}).get(metric, {})
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
        return float(value["mean"])
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _pct(gil_value: float | None, nogil_value: float | None) -> str:
    if gil_value is None or nogil_value is None or gil_value == 0.0:
        return "n/a"
    return f"{((nogil_value / gil_value) - 1.0) * 100.0:+.1f}%"


def _build_comparison(gil_summary: dict[str, Any], nogil_summary: dict[str, Any]) -> dict[str, Any]:
    gil_variants = _variant_by_name(gil_summary)
    nogil_variants = _variant_by_name(nogil_summary)
    rows = []
    for variant_name in sorted(set(gil_variants) & set(nogil_variants)):
        for metric in METRICS:
            gil_mean = _mean(gil_variants[variant_name], metric)
            nogil_mean = _mean(nogil_variants[variant_name], metric)
            rows.append(
                {
                    "variant": variant_name,
                    "metric": metric,
                    "gil_mean": gil_mean,
                    "nogil_mean": nogil_mean,
                    "delta": (
                        nogil_mean - gil_mean
                        if gil_mean is not None and nogil_mean is not None
                        else None
                    ),
                    "nogil_vs_gil_percent": (
                        ((nogil_mean / gil_mean) - 1.0) * 100.0
                        if gil_mean not in (None, 0.0) and nogil_mean is not None
                        else None
                    ),
                }
            )
    return {
        "gil": gil_summary,
        "nogil": nogil_summary,
        "rows": rows,
    }


def _print_table(comparison: dict[str, Any]) -> None:
    print("\nComparison means; negative delta means no-GIL was faster/lower.")
    print("| Variant | Metric | GIL | no-GIL | Delta | no-GIL vs GIL |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in comparison["rows"]:
        gil_mean = row["gil_mean"]
        nogil_mean = row["nogil_mean"]
        delta = row["delta"]
        print(
            f"| {row['variant']} | {row['metric']} | "
            f"{_fmt(gil_mean)} | {_fmt(nogil_mean)} | {_fmt(delta)} | "
            f"{_pct(gil_mean, nogil_mean)} |"
        )


def main() -> None:
    args = _parse_args()
    benchmark_args = _strip_separator(list(args.benchmark_args))
    python_path = Path(args.python)
    if not python_path.exists():
        raise SystemExit(f"Python executable not found: {python_path}")

    output_dir = Path(args.output_dir)
    command = _benchmark_command(args, benchmark_args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("GIL command:")
        print("PYTHON_GIL=1 " + " ".join(command))
        print("no-GIL command:")
        print("PYTHON_GIL=0 " + " ".join(command))
        print(f"Output directory: {output_dir}")
        return

    gil_summary = _run_mode(
        label="gil",
        gil_value="1",
        command=command,
        output_dir=output_dir,
    )
    nogil_summary = _run_mode(
        label="nogil",
        gil_value="0",
        command=command,
        output_dir=output_dir,
    )
    comparison = _build_comparison(gil_summary, nogil_summary)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _print_table(comparison)
    print(f"\nWrote logs and summaries to: {output_dir}")


if __name__ == "__main__":
    main()
