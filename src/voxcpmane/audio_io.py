"""Numpy-only audio I/O.

Drop-in replacements for the small subset of librosa that VoxCPM's
``_encode_wav`` needs, built on ``soundfile`` (libsndfile) for decoding and
``soxr`` for high-quality resampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import soxr


def load_audio(
    path: Path,
    target_sr: int | None = None,
    mono: bool = True,
    dtype: str = "float32",
    resample_quality: str = "HQ",
) -> Tuple[np.ndarray, int]:
    """Load an audio file, optionally resampling to ``target_sr``.

    Args:
        path: audio file path.
        target_sr: target sample rate. If ``None``, keep the file's native
            rate. Otherwise the signal is resampled via ``soxr`` at the
            requested quality preset.
        mono: if ``True`` and the file is multichannel, channels are averaged
            to produce a single-channel signal. If ``False``, a
            ``[channels, samples]`` array is returned.
        dtype: numpy dtype of the returned array. Must be one of the float
            dtypes supported by ``soundfile`` (``float32`` or ``float64``).
        resample_quality: soxr quality preset. ``"HQ"`` (default) is a good
            speed/quality tradeoff; ``"VHQ"`` is higher quality but slower.

    Returns:
        ``(audio, sample_rate)``. ``audio`` is a 1-D array of length
        ``num_samples`` when ``mono=True``, else a 2-D array of shape
        ``(num_channels, num_samples)``. ``sample_rate`` is the rate of the
        returned signal (equal to ``target_sr`` if resampled).
    """
    audio, sr = sf.read(str(path), dtype=dtype, always_2d=False)

    if audio.ndim > 1:
        if mono:
            audio = audio.mean(axis=-1)
        else:
            audio = audio.T

    if target_sr is not None and sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr, quality=resample_quality)
        sr = target_sr

    return audio.astype(dtype, copy=False), sr
