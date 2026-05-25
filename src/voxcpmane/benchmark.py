"""Benchmark baseline vs async-overlap VoxCPM2 generation variants."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import server


@dataclass(frozen=True)
class Variant:
    name: str
    flags: dict[str, Any]


def _build_request(args: argparse.Namespace) -> server.SpeechRequest:
    return server.SpeechRequest(
        model="voxcpm2",
        input=args.text,
        voice=args.voice,
        voice_mode=args.voice_mode,
        reference_wav_path=args.reference_wav_path,
        prompt_wav_path=args.prompt_wav_path,
        prompt_text=args.prompt_text,
        max_length=args.max_length,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        seed=args.seed,
        normalize=args.normalize,
    )


def _parse_token_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("["):
        parsed = json.loads(text)
        return [int(item) for item in parsed]
    return [int(part) for part in text.replace(",", " ").split()]


def _summarize_decode_ready(events: list[dict[str, Any]]) -> dict[str, Any]:
    waits: dict[str, float] = {}
    total_without_labels = 0.0
    for event in events:
        if event["kind"] != "decode_ready":
            continue
        values = event["values"]
        by_lm = values.get("decode_load_wait_by_lm")
        if isinstance(by_lm, dict):
            for name, seconds in by_lm.items():
                waits[str(name)] = float(seconds)
        elif values.get("decode_load_wait_seconds") is not None:
            total_without_labels += float(values["decode_load_wait_seconds"])
    total = sum(waits.values()) + total_without_labels
    return {
        "decode_load_wait_seconds": total if total > 0.0 or waits else None,
        "decode_load_wait_by_lm": waits,
    }


def _seconds_per_audio(second_count: float, audio_seconds: float) -> float | None:
    if second_count <= 0.0 or audio_seconds <= 0.0:
        return None
    return second_count / audio_seconds


def _load_variant(args: argparse.Namespace, variant: Variant) -> None:
    load_kwargs = {
        "model_dir": args.model_dir,
        "repo_id": args.repo_id,
        "included_voice_cache_dir": args.included_voice_cache_dir,
        "embedding_path": args.embedding_path,
        "lm_mode": args.lm_mode,
        "lm_prefill_chunk_size": args.lm_prefill_chunk_size,
        "base_lm_splits": args.base_lm_splits,
        "compiled_fallback_dir": args.compiled_fallback_dir,
        "vae_early_decode_steps": args.vae_early_decode_steps,
        "vae_batch_decode_steps": args.vae_batch_decode_steps,
        "base_lm_path": args.base_lm_path,
        "residual_lm_path": args.residual_lm_path,
        "locdit_path": args.locdit_path,
        "vae_encoder_path": args.vae_encoder_path,
        "feat_encoder_path": args.feat_encoder_path,
        "vae_decoder_path": args.vae_decoder_path,
        "fsq_path": args.fsq_path,
        "projections_path": args.projections_path,
        "compile_and_save": args.compile_and_save,
        "preload_tokenizer": _parse_token_ids(args.token_ids) is None,
    }
    load_kwargs.update(variant.flags)
    server.load_model(**load_kwargs)


def _run_once(request: server.SpeechRequest) -> dict[str, Any]:
    events: list[dict[str, Any]] = []

    def record_metric(kind: str, values: dict) -> None:
        events.append({"kind": kind, "values": values})

    started = time.perf_counter()
    first_chunk_at = None
    chunks = 0
    samples = 0
    for chunk in server.generate_audio_chunks(
        request,
        metrics_callback=record_metric,
    ):
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter()
        arr = np.asarray(chunk)
        chunks += 1
        samples += int(arr.shape[0])
    finished = time.perf_counter()

    by_kind = {event["kind"]: event["values"] for event in events}
    prefill = by_kind.get("prefill", {})
    generation_start = by_kind.get("generation_start", {})
    final = by_kind.get("final", {})
    decode_ready = _summarize_decode_ready(events)
    cleanup = by_kind.get("prefill_cleanup", {})
    first_ar_iteration = by_kind.get("first_ar_iteration", {})
    audio_seconds = samples / float(server.SAMPLE_RATE) if samples else 0.0
    wall_seconds = finished - started
    generation_seconds = float(final.get("generation_seconds", 0.0) or 0.0)
    decode_wait_seconds = float(decode_ready.get("decode_load_wait_seconds") or 0.0)
    generation_excluding_decode_wait = max(
        0.0,
        generation_seconds - decode_wait_seconds,
    )
    first_ar_iteration_at = first_ar_iteration.get("at")
    return {
        "wall_seconds": wall_seconds,
        "ttfb_seconds": (
            first_chunk_at - started if first_chunk_at is not None else None
        ),
        "first_ar_iteration_wall_seconds": (
            float(first_ar_iteration_at) - started
            if isinstance(first_ar_iteration_at, (int, float))
            else None
        ),
        "prefill_seconds": prefill.get("prefill_seconds"),
        "prefill_stage_seconds": prefill.get("prefill_stage_seconds", {}),
        "swap_to_decode_seconds": generation_start.get("swap_to_decode_seconds"),
        "decode_load_wait_seconds": decode_ready.get("decode_load_wait_seconds"),
        "decode_load_wait_by_lm": decode_ready.get("decode_load_wait_by_lm", {}),
        "prefill_cleanup_wait_seconds": cleanup.get("prefill_cleanup_wait_seconds"),
        "first_ar_iteration_seconds": first_ar_iteration.get(
            "first_ar_iteration_seconds"
        ),
        "generation_seconds": generation_seconds,
        "generation_seconds_excluding_decode_wait": generation_excluding_decode_wait,
        "audio_seconds": audio_seconds,
        "rtf": _seconds_per_audio(generation_seconds, audio_seconds),
        "rtf_excluding_decode_wait": _seconds_per_audio(
            generation_excluding_decode_wait,
            audio_seconds,
        ),
        "chunks": chunks,
        "samples": samples,
        "events": events,
    }


def _direct_gen_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    target_token_ids = _parse_token_ids(args.token_ids)
    if target_token_ids is None:
        raise ValueError("--token-ids is required for direct token-ID benchmark mode")
    prompt_token_ids = _parse_token_ids(args.prompt_token_ids)
    gen_kwargs: dict[str, Any] = {
        "target_text": args.text,
        "target_token_ids": target_token_ids,
        "prompt_token_ids": prompt_token_ids,
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
        "max_len": args.max_length,
        "seed": args.seed,
    }
    reference_audio_embed = None
    prompt_audio_embed = None
    prompt_prefix_feat_cond = None
    prompt_decode_context = None
    voice_name = (args.voice or "").strip() or None
    if voice_name is not None:
        voice_name = server.VOICE_STORE.validate(voice_name)
        voice_mode = (args.voice_mode or "reference").strip().replace("-", "_")
        reference_audio_embed = server.load_voice_feature_cache(voice_name).astype(
            np.float32,
            copy=False,
        )
        read_paths, cache_path = server.lm_prefix_cache_paths(voice_name)
        gen_kwargs["lm_prefix_cache_read_paths"] = read_paths
        gen_kwargs["lm_prefix_cache_path"] = cache_path
        if voice_mode == "high_similarity":
            prompt_audio_embed = server.load_voice_feature_cache(
                voice_name,
                prompt=True,
            ).astype(np.float32, copy=False)
            prompt_prefix_feat_cond = server.load_voice_prompt_cond(voice_name)
            prompt_decode_context = server.load_voice_prompt_decode_context(voice_name)
            gen_kwargs["prompt_text"] = args.prompt_text
        elif voice_mode not in {"reference", "reference_plus_prompt"}:
            raise ValueError(f"unsupported voice mode for benchmark: {voice_mode}")
    if args.reference_wav_path:
        gen_kwargs["reference_wav_path"] = args.reference_wav_path
    elif reference_audio_embed is not None:
        gen_kwargs["reference_audio_embed"] = reference_audio_embed
    if args.prompt_wav_path:
        gen_kwargs["prompt_wav_path"] = args.prompt_wav_path
        gen_kwargs["prompt_text"] = args.prompt_text
    elif prompt_audio_embed is not None:
        gen_kwargs["prompt_audio_embed"] = prompt_audio_embed
        gen_kwargs["prompt_prefix_feat_cond"] = prompt_prefix_feat_cond
        gen_kwargs["prompt_decode_context"] = prompt_decode_context
    return gen_kwargs


def _run_once_direct(gen_kwargs: dict[str, Any]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []

    def record_metric(kind: str, values: dict) -> None:
        events.append({"kind": kind, "values": values})

    started = time.perf_counter()
    first_chunk_at = None
    chunks = 0
    samples = 0
    for chunk in server.generator.generate_streaming(
        **gen_kwargs,
        metrics_callback=record_metric,
    ):
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter()
        arr = np.asarray(chunk)
        chunks += 1
        samples += int(arr.shape[0])
    finished = time.perf_counter()

    by_kind = {event["kind"]: event["values"] for event in events}
    prefill = by_kind.get("prefill", {})
    generation_start = by_kind.get("generation_start", {})
    final = by_kind.get("final", {})
    decode_ready = _summarize_decode_ready(events)
    cleanup = by_kind.get("prefill_cleanup", {})
    first_ar_iteration = by_kind.get("first_ar_iteration", {})
    audio_seconds = samples / float(server.SAMPLE_RATE) if samples else 0.0
    generation_seconds = float(final.get("generation_seconds", 0.0) or 0.0)
    decode_wait_seconds = float(decode_ready.get("decode_load_wait_seconds") or 0.0)
    generation_excluding_decode_wait = max(
        0.0,
        generation_seconds - decode_wait_seconds,
    )
    first_ar_iteration_at = first_ar_iteration.get("at")
    return {
        "wall_seconds": finished - started,
        "ttfb_seconds": (
            first_chunk_at - started if first_chunk_at is not None else None
        ),
        "first_ar_iteration_wall_seconds": (
            float(first_ar_iteration_at) - started
            if isinstance(first_ar_iteration_at, (int, float))
            else None
        ),
        "prefill_seconds": prefill.get("prefill_seconds"),
        "prefill_stage_seconds": prefill.get("prefill_stage_seconds", {}),
        "swap_to_decode_seconds": generation_start.get("swap_to_decode_seconds"),
        "decode_load_wait_seconds": decode_ready.get("decode_load_wait_seconds"),
        "decode_load_wait_by_lm": decode_ready.get("decode_load_wait_by_lm", {}),
        "prefill_cleanup_wait_seconds": cleanup.get("prefill_cleanup_wait_seconds"),
        "first_ar_iteration_seconds": first_ar_iteration.get(
            "first_ar_iteration_seconds"
        ),
        "generation_seconds": generation_seconds,
        "generation_seconds_excluding_decode_wait": generation_excluding_decode_wait,
        "audio_seconds": audio_seconds,
        "rtf": _seconds_per_audio(generation_seconds, audio_seconds),
        "rtf_excluding_decode_wait": _seconds_per_audio(
            generation_excluding_decode_wait,
            audio_seconds,
        ),
        "chunks": chunks,
        "samples": samples,
        "events": events,
    }


def _summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "wall_seconds",
        "ttfb_seconds",
        "first_ar_iteration_wall_seconds",
        "prefill_seconds",
        "swap_to_decode_seconds",
        "decode_load_wait_seconds",
        "first_ar_iteration_seconds",
        "generation_seconds",
        "generation_seconds_excluding_decode_wait",
        "audio_seconds",
        "rtf",
        "rtf_excluding_decode_wait",
    )
    summary: dict[str, Any] = {"runs": len(values)}
    for key in keys:
        nums = [run[key] for run in values if isinstance(run.get(key), (int, float))]
        if not nums:
            continue
        summary[key] = {
            "mean": statistics.fmean(nums),
            "min": min(nums),
            "max": max(nums),
        }
    return summary


def _variants(args: argparse.Namespace) -> list[Variant]:
    async_flags = {
        "lm_async_decode_load": True,
        "lm_async_prefill_unload": True,
        "prefill_audio_async": True,
        "prefill_audio_queue_size": args.prefill_audio_queue_size,
        "vae_async_decode": True,
        "vae_decode_max_pending": args.vae_decode_max_pending,
    }
    selected = []
    if args.variant in {"baseline", "both"}:
        selected.append(Variant("baseline", {}))
    if args.variant in {"async", "both"}:
        selected.append(Variant("async", async_flags))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VoxCPM2 baseline vs async overlap variants."
    )
    parser.add_argument("--variant", choices=("baseline", "async", "both"), default="both")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--text", default="Hello from VoxCPM2 on Apple Neural Engine.")
    parser.add_argument(
        "--token-ids",
        default=None,
        help=(
            "Comma/space-separated target token IDs, or a JSON integer array. "
            "When set, the benchmark bypasses tokenizers so it can "
            "run in minimal Python 3.14t environments."
        ),
    )
    parser.add_argument(
        "--prompt-token-ids",
        default=None,
        help="Token IDs for prompt_text when benchmarking prompt audio with --token-ids.",
    )
    parser.add_argument("--voice", default=None)
    parser.add_argument("--voice-mode", default="reference")
    parser.add_argument("--reference-wav-path", default=None)
    parser.add_argument("--prompt-wav-path", default=None)
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--repo-id", default=server.REPO_ID)
    parser.add_argument("--included-voice-cache-dir", default=None)
    parser.add_argument("--embedding-path", default=None)
    parser.add_argument("--lm-mode", default="hot-swap")
    parser.add_argument("--lm-prefill-chunk-size", type=int, default=128)
    parser.add_argument("--base-lm-splits", type=int, default=2)
    parser.add_argument("--compiled-fallback-dir", default=None)
    parser.add_argument("--vae-early-decode-steps", type=int, default=1)
    parser.add_argument("--vae-batch-decode-steps", type=int, default=1)
    parser.add_argument("--prefill-audio-queue-size", type=int, default=2)
    parser.add_argument("--vae-decode-max-pending", type=int, default=2)
    parser.add_argument("--base-lm-path", nargs="+", default=None)
    for option in (
        "residual-lm-path",
        "locdit-path",
        "vae-encoder-path",
        "feat-encoder-path",
        "vae-decoder-path",
        "fsq-path",
        "projections-path",
    ):
        parser.add_argument(f"--{option}", default=None)
    parser.add_argument("--compile-and-save", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = None if args.token_ids else _build_request(args)
    output = {
        "python": {
            "implementation": sys.implementation.name,
            "version": sys.version,
            "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: None)(),
        },
        "variants": [],
    }
    for variant in _variants(args):
        _load_variant(args, variant)
        gen_kwargs = _direct_gen_kwargs(args) if args.token_ids else None
        runs = []
        for index in range(args.warmup_runs + args.runs):
            result = (
                _run_once_direct(gen_kwargs)
                if gen_kwargs is not None
                else _run_once(request)
            )
            result["warmup"] = index < args.warmup_runs
            result["index"] = index
            print(json.dumps({"variant": variant.name, **result}, default=str), flush=True)
            if not result["warmup"]:
                runs.append(result)
        output["variants"].append({
            "name": variant.name,
            "flags": variant.flags,
            "summary": _summarize(runs),
        })
        gc.collect()
    print(json.dumps({"summary": output}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
