"""Metrics and RTF statistics helpers for VoxCPM2 generation jobs."""

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

LIVE_RTF_METRICS = "off"
SAMPLE_RATE = 48000


@dataclass
class GenerationJob:
    request: "SpeechRequest"
    output_queue: queue.Queue
    cancel_event: threading.Event
    job_id: int
    created_at: float = field(default_factory=time.perf_counter)
    prefill_seconds: Optional[float] = None
    prefill_tokens: Optional[int] = None
    prefill_stage_seconds: dict[str, float] = field(default_factory=dict)
    prefill_text_tokens: Optional[int] = None
    prefill_audio_tokens: Optional[int] = None
    prefill_reference_audio_tokens: Optional[int] = None
    prefill_prompt_audio_tokens: Optional[int] = None
    prefill_lm_prefix_cache: Optional[bool] = None
    prefill_lm_prefix_tokens: Optional[int] = None
    prefill_done_at: Optional[float] = None
    swap_to_decode_seconds: Optional[float] = None
    generation_started_at: Optional[float] = None
    generation_seconds: Optional[float] = None
    audio_seconds: Optional[float] = None
    ttfb_seconds: Optional[float] = None
    post_prefill_ttfb_seconds: Optional[float] = None
    first_loop_seconds: Optional[float] = None
    status: Optional[str] = None
    first_byte_printed: bool = False
    final_printed: bool = False
    audio_samples_sent: int = 0
    inference_loops_sent: int = 0
    live_rtf_started: bool = False
    last_live_rtf_at: Optional[float] = None


@dataclass
class GenerationMetricEvent:
    kind: str
    values: dict


def _metric_value(value: Optional[float], suffix: str = "s") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def _prefill_tokens_per_second(job: GenerationJob) -> Optional[float]:
    if not job.prefill_seconds or not job.prefill_tokens:
        return None
    return job.prefill_tokens / job.prefill_seconds


def _prefill_token_breakdown(job: GenerationJob) -> str:
    if (
        job.prefill_text_tokens is None
        and job.prefill_audio_tokens is None
        and job.prefill_reference_audio_tokens is None
        and job.prefill_prompt_audio_tokens is None
    ):
        return ""
    parts = [
        f"text={job.prefill_text_tokens if job.prefill_text_tokens is not None else 'n/a'}",
        f"audio={job.prefill_audio_tokens if job.prefill_audio_tokens is not None else 'n/a'}",
        f"ref_audio={job.prefill_reference_audio_tokens if job.prefill_reference_audio_tokens is not None else 'n/a'}",
        f"prompt_audio={job.prefill_prompt_audio_tokens if job.prefill_prompt_audio_tokens is not None else 'n/a'}",
        f"total={job.prefill_tokens if job.prefill_tokens is not None else 'n/a'}",
    ]
    if job.prefill_lm_prefix_tokens:
        cache_state = "hit" if job.prefill_lm_prefix_cache else "miss"
        parts.append(f"lm_prefix_cache={cache_state}:{job.prefill_lm_prefix_tokens}")
    return "tokens[" + " ".join(parts) + "]"


def _prefill_stage_breakdown(job: GenerationJob) -> str:
    if not job.prefill_stage_seconds:
        return ""
    stage_labels = (
        ("text_tokens", "text"),
        ("audio_features", "audio"),
        ("sequence_build", "seq"),
        ("text_embed", "text_emb"),
        ("feat_embed", "feat_emb"),
        ("combine", "combine"),
        ("lm_prefix_cache_restore", "lm_cache_restore"),
        ("base_lm", "base_lm"),
        ("fsq", "fsq"),
        ("residual_prep", "res_prep"),
        ("residual_lm", "res_lm"),
        ("lm_prefix_cache_save", "lm_cache_save"),
    )
    parts = [
        f"{label}={_metric_value(job.prefill_stage_seconds[name])}"
        for name, label in stage_labels
        if name in job.prefill_stage_seconds
    ]
    extras = [
        f"{name}={_metric_value(value)}"
        for name, value in sorted(job.prefill_stage_seconds.items())
        if name not in {stage_name for stage_name, _ in stage_labels}
    ]
    return "stages[" + " ".join(parts + extras) + "]"


def _print_prefill_detail_metrics(job: GenerationJob) -> None:
    details = [
        detail
        for detail in (_prefill_token_breakdown(job), _prefill_stage_breakdown(job))
        if detail
    ]
    if not details:
        return
    print(
        f"⏱ Job {job.job_id}: prefill_detail " + " ".join(details),
        flush=True,
    )


def _print_first_byte_metrics(job: GenerationJob) -> None:
    _finish_live_rtf_line(job)
    rate = _prefill_tokens_per_second(job)
    rate_text = "n/a" if rate is None else f"{rate:.1f} tok/s"
    print(
        f"⏱ Job {job.job_id}: first_byte "
        f"prefill={_metric_value(job.prefill_seconds)} "
        f"swap_to_decode={_metric_value(job.swap_to_decode_seconds)} "
        f"prompt_tps={rate_text} "
        f"ttfb={_metric_value(job.ttfb_seconds)} "
        f"post_prefill_ttfb={_metric_value(job.post_prefill_ttfb_seconds)} "
        f"first_loop={_metric_value(job.first_loop_seconds)}",
        flush=True,
    )
    _print_prefill_detail_metrics(job)


def _print_final_metrics(job: GenerationJob, status: Optional[str] = None) -> None:
    if job.final_printed:
        return
    _finish_live_rtf_line(job)
    if status is not None:
        job.status = status
    status_text = job.status or "completed"
    rate = _prefill_tokens_per_second(job)
    rate_text = "n/a" if rate is None else f"{rate:.1f} tok/s"
    audio_seconds = job.audio_seconds
    if audio_seconds is None and job.audio_samples_sent:
        audio_seconds = job.audio_samples_sent / float(SAMPLE_RATE)
    generation_seconds = job.generation_seconds
    if generation_seconds is None and job.generation_started_at is not None:
        generation_seconds = time.perf_counter() - job.generation_started_at
    rtf = (
        generation_seconds / audio_seconds
        if generation_seconds is not None and audio_seconds and audio_seconds > 0.0
        else None
    )
    rtf_text = "n/a" if rtf is None else f"{rtf:.3f}"
    avg_loop_period = (
        generation_seconds / job.inference_loops_sent
        if generation_seconds and generation_seconds > 0.0 and job.inference_loops_sent
        else None
    )
    loop_text = "n/a" if avg_loop_period is None else f"{avg_loop_period:.3f}s"
    print(
        f"⏱ Job {job.job_id}: {status_text} "
        f"prefill={_metric_value(job.prefill_seconds)} "
        f"swap_to_decode={_metric_value(job.swap_to_decode_seconds)} "
        f"prompt_tps={rate_text} "
        f"ttfb={_metric_value(job.ttfb_seconds)} "
        f"post_prefill_ttfb={_metric_value(job.post_prefill_ttfb_seconds)} "
        f"first_loop={_metric_value(job.first_loop_seconds)} "
        f"rtf={rtf_text} "
        f"loop_period={loop_text} "
        f"audio={_metric_value(audio_seconds)} "
        f"inference={_metric_value(generation_seconds)}",
        flush=True,
    )
    _print_prefill_detail_metrics(job)
    job.final_printed = True


def _handle_metric_event(job: GenerationJob, event: GenerationMetricEvent) -> None:
    values = event.values
    if event.kind == "prefill":
        job.prefill_seconds = float(values.get("prefill_seconds", 0.0))
        job.prefill_tokens = int(values.get("prefill_tokens", 0))
        job.prefill_stage_seconds = {
            str(name): float(seconds)
            for name, seconds in values.get("prefill_stage_seconds", {}).items()
        }
        text_tokens = values.get("prefill_text_tokens")
        audio_tokens = values.get("prefill_audio_tokens")
        reference_audio_tokens = values.get("prefill_reference_audio_tokens")
        prompt_audio_tokens = values.get("prefill_prompt_audio_tokens")
        if text_tokens is not None:
            job.prefill_text_tokens = int(text_tokens)
        if audio_tokens is not None:
            job.prefill_audio_tokens = int(audio_tokens)
        if reference_audio_tokens is not None:
            job.prefill_reference_audio_tokens = int(reference_audio_tokens)
        if prompt_audio_tokens is not None:
            job.prefill_prompt_audio_tokens = int(prompt_audio_tokens)
        lm_prefix_cache = values.get("prefill_lm_prefix_cache")
        lm_prefix_tokens = values.get("prefill_lm_prefix_tokens")
        if lm_prefix_cache is not None:
            job.prefill_lm_prefix_cache = bool(lm_prefix_cache)
        if lm_prefix_tokens is not None:
            job.prefill_lm_prefix_tokens = int(lm_prefix_tokens)
        job.prefill_done_at = float(values.get("at", time.perf_counter()))
    elif event.kind == "generation_start":
        job.generation_started_at = float(values.get("at", time.perf_counter()))
        swap_seconds = values.get("swap_to_decode_seconds")
        if swap_seconds is not None:
            job.swap_to_decode_seconds = float(swap_seconds)
    elif event.kind == "final":
        job.status = str(values.get("status", "completed"))
        job.generation_seconds = float(values.get("generation_seconds", 0.0))
        job.audio_seconds = float(values.get("audio_seconds", 0.0))
        if job.ttfb_seconds is not None:
            _print_final_metrics(job)


def _mark_first_byte(job: GenerationJob) -> None:
    if job.first_byte_printed:
        return
    job.ttfb_seconds = time.perf_counter() - job.created_at
    if job.prefill_done_at is not None:
        job.post_prefill_ttfb_seconds = time.perf_counter() - job.prefill_done_at
    if job.generation_started_at is not None:
        job.first_loop_seconds = time.perf_counter() - job.generation_started_at
    _print_first_byte_metrics(job)
    job.first_byte_printed = True
    if job.generation_seconds is not None and not job.final_printed:
        _print_final_metrics(job)


def _finish_live_rtf_line(job: GenerationJob) -> None:
    if not job.live_rtf_started:
        return
    sys.stdout.write("\n")
    sys.stdout.flush()
    job.live_rtf_started = False


def _compute_rtf_stats(job: GenerationJob, chunk_samples: int) -> Optional[dict]:
    """Compute RTF statistics for the current chunk. Returns None if not ready."""
    if chunk_samples <= 0 or job.generation_started_at is None:
        return None
    now = time.perf_counter()
    audio_seconds = job.audio_samples_sent / float(SAMPLE_RATE)
    if audio_seconds <= 0.0:
        return None
    generation_elapsed = max(0.0, now - job.generation_started_at)
    avg_rtf = generation_elapsed / audio_seconds
    chunk_audio_seconds = chunk_samples / float(SAMPLE_RATE)
    previous = job.last_live_rtf_at or job.generation_started_at
    chunk_elapsed = max(0.0, now - previous)
    inst_rtf = chunk_elapsed / chunk_audio_seconds
    avg_loop_period = (
        generation_elapsed / job.inference_loops_sent
        if generation_elapsed > 0.0 and job.inference_loops_sent
        else 0.0
    )
    job.last_live_rtf_at = now
    loop_label = "first_loop" if job.inference_loops_sent == 1 else "loop_period"
    return {
        "avg_rtf": avg_rtf,
        "inst_rtf": inst_rtf,
        "avg_loop_period": avg_loop_period,
        "loop_label": loop_label,
        "chunk_elapsed": chunk_elapsed,
        "audio_seconds": audio_seconds,
        "generation_elapsed": generation_elapsed,
    }


def _update_live_rtf(job: GenerationJob, chunk_samples: int) -> None:
    if LIVE_RTF_METRICS != "live" or chunk_samples <= 0:
        return
    stats = _compute_rtf_stats(job, chunk_samples)
    if stats is None:
        return
    job.live_rtf_started = True
    sys.stdout.write(
        f"\r⏱ Job {job.job_id}: rtf_avg={stats['avg_rtf']:.3f} "
        f"rtf_inst={stats['inst_rtf']:.3f} loop_avg={stats['avg_loop_period']:.3f}s "
        f"{stats['loop_label']}={stats['chunk_elapsed']:.3f}s audio={stats['audio_seconds']:.2f}s "
        f"inference={stats['generation_elapsed']:.2f}s"
    )
    sys.stdout.flush()


def _update_final_rtf(job: GenerationJob, chunk_samples: int) -> None:
    """Track stats for final-mode RTF display (no live printing)."""
    if LIVE_RTF_METRICS != "final" or chunk_samples <= 0:
        return
    _compute_rtf_stats(job, chunk_samples)


def _print_final_rtf_summary(job: GenerationJob) -> None:
    """Print the final RTF summary line when --live-rtf=final."""
    if LIVE_RTF_METRICS != "final":
        return
    if job.generation_started_at is None or not job.audio_samples_sent:
        return
    now = time.perf_counter()
    audio_seconds = job.audio_samples_sent / float(SAMPLE_RATE)
    if audio_seconds <= 0.0:
        return
    generation_elapsed = max(0.0, now - job.generation_started_at)
    avg_rtf = generation_elapsed / audio_seconds
    avg_loop_period = (
        generation_elapsed / job.inference_loops_sent
        if generation_elapsed > 0.0 and job.inference_loops_sent
        else 0.0
    )
    print(
        f"⏱ Job {job.job_id}: rtf_avg={avg_rtf:.3f} "
        f"loop_avg={avg_loop_period:.3f}s "
        f"audio={audio_seconds:.2f}s "
        f"inference={generation_elapsed:.2f}s",
        flush=True,
    )
