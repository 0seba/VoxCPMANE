"""Numpy / CoreML inference wrapper for the VoxCPM2 AudioVAE encoder.

Runtime counterpart to ``src/qeml/conversion/voxcpm2/audio_vae_encoder.py``.

The converted ``.mlpackage`` has a **fixed per-call input length** and
threads a per-stage **streaming cache** through every call. Callers are
expected to keep the encoder instance alive across chunks so that the
cache is carried forward; :meth:`AudioVAEEncoder.reset` re-zeros it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import coremltools as ct
import numpy as np

from ._coreml_utils import (
    discover_cache_shapes,
    load_coreml_model,
)
from .audio_io import load_audio

# Defaults matching AudioVAEConfig in upstream VoxCPM 2.
DEFAULT_HOP_LENGTH = 640  # math.prod([2, 5, 8, 8])
DEFAULT_LATENT_DIM = 64
DEFAULT_PATCH_SIZE = 4
DEFAULT_SAMPLE_RATE = 16000


class AudioVAEEncoder:
    """Stateful wrapper around a converted AudioVAE encoder ``.mlpackage``.

    The wrapped CoreML model expects a fixed-size input window
    (``chunk_samples``) and takes/returns one cache tensor per encoder
    stage. This class hides the cache plumbing: call :meth:`encode_chunk`
    with successive chunks and the cache is threaded through
    automatically. :meth:`reset` zeroes the cache between independent
    audio streams. :meth:`encode_wav` is a drop-in replacement for
    ``VoxCPM2Model._encode_wav`` that handles loading, resampling,
    padding, and chunked streaming end to end.
    """

    def __init__(
        self,
        model_path: Path,
        chunk_samples: int,
        hop_length: int = DEFAULT_HOP_LENGTH,
        latent_dim: int = DEFAULT_LATENT_DIM,
        patch_size: int = DEFAULT_PATCH_SIZE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_ONLY,
    ):
        self.model = load_coreml_model(model_path, compute_units=compute_units)
        self.chunk_samples = chunk_samples
        self.hop_length = hop_length
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.sample_rate = sample_rate

        self.patch_len = self.patch_size * self.hop_length
        if self.chunk_samples % self.hop_length != 0:
            raise ValueError(
                f"chunk_samples ({self.chunk_samples}) must be a multiple of "
                f"hop_length ({self.hop_length})"
            )
        self.frames_per_chunk = self.chunk_samples // self.hop_length

        self._cache_shapes = discover_cache_shapes(self.model, model_path)
        self._caches: List[np.ndarray] = []
        self.reset()

    # ------------------------------------------------------------------ #
    # streaming API
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Re-zero the per-stage cache tensors."""
        self._caches = [np.zeros(s, dtype=np.float32) for s in self._cache_shapes]

    @property
    def cache_shapes(self) -> List[Tuple[int, ...]]:
        return self._cache_shapes

    def encode_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Encode one fixed-size audio chunk, advancing the cache.

        Args:
            audio_chunk: 1-D float32 array of length ``chunk_samples``.

        Returns:
            Latent frames of shape ``(1, latent_dim, frames_per_chunk)``.
        """
        inputs = {
            "audio": np.ascontiguousarray(
                audio_chunk.reshape(1, 1, self.chunk_samples), dtype=np.float32
            )
        }
        for i, c in enumerate(self._caches):
            inputs[f"cache_{i}"] = c

        result = self.model.predict(inputs)

        self._caches = [
            result[f"new_cache_{i}"]
            for i in range(len(self._caches))
        ]
        return result["mu"]

    def encode_patches_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Encode one chunk into VoxCPM-style patches.

        Returns:
            Array of shape ``(frames_per_chunk // patch_size, patch_size,
            latent_dim)``.
        """
        mu = self.encode_chunk(audio_chunk)  # (1, D, F)
        mu = mu.reshape(self.latent_dim, -1, self.patch_size)
        return np.transpose(mu, (1, 2, 0))

    # ------------------------------------------------------------------ #
    # high-level _encode_wav replacement
    # ------------------------------------------------------------------ #
    def encode_wav(
        self,
        wav_path: Path,
        padding_mode: str = "right",
    ) -> np.ndarray:
        """Load a WAV file, resample, pad, and encode into VoxCPM patches.

        Pure-numpy replacement for ``VoxCPM2Model._encode_wav``. Resets
        the streaming cache before starting so the result does not depend
        on any previous call. Matches the upstream padding semantics: the
        audio is padded to a multiple of ``patch_len`` (not
        ``chunk_samples``); trailing patches that only exist to pad the
        final chunk to ``chunk_samples`` are dropped.

        Args:
            wav_path: audio file path.
            padding_mode: ``"right"`` (default) or ``"left"`` padding,
                used only to align the audio to ``patch_len``. Matches
                the upstream parameter of the same name.

        Returns:
            Float32 array of shape ``(num_patches, patch_size,
            latent_dim)``.
        """
        audio, _ = load_audio(wav_path, target_sr=self.sample_rate, mono=True)
        audio = self._pad_to_patch_len(audio, padding_mode)
        target_patches = len(audio) // self.patch_len

        audio = self._pad_to_chunk(audio)

        self.reset()
        chunks = [
            self.encode_patches_chunk(audio[start : start + self.chunk_samples])
            for start in range(0, len(audio), self.chunk_samples)
        ]
        all_patches = np.concatenate(chunks, axis=0)
        return all_patches[:target_patches]

    # ------------------------------------------------------------------ #
    # padding helpers
    # ------------------------------------------------------------------ #
    def _pad_to_patch_len(self, audio: np.ndarray, padding_mode: str) -> np.ndarray:
        remainder = len(audio) % self.patch_len
        if remainder == 0:
            return audio
        pad = self.patch_len - remainder
        if padding_mode == "left":
            return np.pad(audio, (pad, 0))
        if padding_mode == "right":
            return np.pad(audio, (0, pad))
        raise ValueError(
            f"padding_mode must be 'left' or 'right', got {padding_mode!r}"
        )

    def _pad_to_chunk(self, audio: np.ndarray) -> np.ndarray:
        remainder = len(audio) % self.chunk_samples
        if remainder == 0:
            return audio
        return np.pad(audio, (0, self.chunk_samples - remainder))
