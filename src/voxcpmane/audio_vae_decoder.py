"""Numpy / CoreML inference wrapper for the VoxCPM2 AudioVAE decoder.

Runtime counterpart to ``src/qeml/conversion/voxcpm2/audio_vae_decoder.py``.
Pure numpy + coremltools.

The converted ``.mlpackage`` has a **fixed per-call latent-frame count**
(``latent_frames``) and threads a per-stage **streaming cache** through
every call. Callers are expected to keep the decoder instance alive
across chunks so that the cache is carried forward;
:meth:`AudioVAEDecoder.reset` re-zeros it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import coremltools as ct
import numpy as np

from ._coreml_utils import (
    discover_cache_shapes,
    load_coreml_model,
)

# Defaults matching AudioVAEConfig in upstream VoxCPM 2.
DEFAULT_LATENT_DIM = 64
DEFAULT_LATENT_FRAMES = 4
DEFAULT_UPSAMPLE_FACTOR = 1920  # math.prod([8, 6, 5, 2, 2, 2])
DEFAULT_PATCH_SIZE = 4
DEFAULT_OUT_SAMPLE_RATE = 48000


class _AudioVAEDecoderBase(ABC):
    """Shared logic for cache-based and MLState-based VAE decoders.

    Subclasses must implement :meth:`decode_chunk` and :meth:`reset`.
    The :meth:`decode_patches` convenience is inherited and calls those
    two methods polymorphically.
    """

    latent_frames: int
    latent_dim: int
    upsample_factor: int
    patch_size: int
    out_sample_rate: int
    samples_per_chunk: int

    @abstractmethod
    def reset(self) -> None:
        """Re-initialise internal state so the next decode is independent."""

    @abstractmethod
    def decode_chunk(self, z_chunk: np.ndarray) -> np.ndarray:
        """Decode one fixed-size latent chunk, advancing internal state."""

    # ------------------------------------------------------------------ #
    # higher-level helpers
    # ------------------------------------------------------------------ #
    def decode_patches(self, patches: np.ndarray) -> np.ndarray:
        """Decode a sequence of VoxCPM patches into a single waveform.

        Accepts the same ``(num_patches, patch_size, latent_dim)`` layout
        produced by the feat decoder / LocDiT + residual model, and
        yields a 1-D float32 waveform. Resets the cache before
        streaming.

        The CoreML graph processes ``latent_frames`` frames per call. If
        ``latent_frames`` does not divide ``num_patches * patch_size``,
        the latent sequence is zero-padded on the right and the
        corresponding trailing audio samples are dropped.
        """
        if patches.ndim != 3:
            raise ValueError(
                f"expected patches of shape (P, S, D), got {patches.shape}"
            )
        num_patches, patch_size, latent_dim = patches.shape
        if latent_dim != self.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: got {latent_dim}, expected {self.latent_dim}"
            )
        if patch_size != self.patch_size:
            raise ValueError(
                f"patch_size mismatch: got {patch_size}, expected {self.patch_size}"
            )

        # (P, S, D) → (D, P*S)
        z = np.transpose(patches, (2, 0, 1)).reshape(
            latent_dim, num_patches * patch_size
        )
        total_frames = z.shape[-1]

        # Pad up to a multiple of latent_frames.
        remainder = total_frames % self.latent_frames
        if remainder:
            pad = self.latent_frames - remainder
            z = np.pad(z, ((0, 0), (0, pad)))

        self.reset()
        audio_chunks: List[np.ndarray] = []
        for start in range(0, z.shape[-1], self.latent_frames):
            chunk = z[:, start : start + self.latent_frames].reshape(
                1, latent_dim, self.latent_frames
            )
            audio_chunks.append(self.decode_chunk(chunk).reshape(-1))
        waveform = np.concatenate(audio_chunks, axis=0)

        keep_samples = total_frames * self.upsample_factor
        return waveform[:keep_samples]


class AudioVAEDecoder(_AudioVAEDecoderBase):
    """Stateful wrapper around a converted AudioVAE decoder ``.mlpackage``.

    The wrapped CoreML model expects a fixed-size latent input window
    (``latent_frames``) and takes/returns one cache tensor per decoder
    stage. This class hides the cache plumbing: call :meth:`decode_chunk`
    with successive latent windows and the cache is threaded through
    automatically. :meth:`reset` zeroes the cache between independent
    streams.

    :meth:`decode_patches` is a convenience that accepts the VoxCPM-style
    ``(num_patches, patch_size, latent_dim)`` layout (what ``LocDiT`` +
    the feat decoder produce) and returns a single waveform, matching
    the behavior of :class:`voxcpm.modules.audiovae.StreamingVAEDecoder`
    iterated over all patches concatenated.
    """

    def __init__(
        self,
        model_path: Path,
        latent_frames: int = DEFAULT_LATENT_FRAMES,
        latent_dim: int = DEFAULT_LATENT_DIM,
        upsample_factor: int = DEFAULT_UPSAMPLE_FACTOR,
        patch_size: int = DEFAULT_PATCH_SIZE,
        out_sample_rate: int = DEFAULT_OUT_SAMPLE_RATE,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_ONLY,
    ):
        self.model = load_coreml_model(model_path, compute_units=compute_units)
        self.latent_frames = int(latent_frames)
        self.latent_dim = int(latent_dim)
        self.upsample_factor = int(upsample_factor)
        self.patch_size = int(patch_size)
        self.out_sample_rate = int(out_sample_rate)

        self.samples_per_chunk = self.latent_frames * self.upsample_factor

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
        return list(self._cache_shapes)

    def decode_chunk(self, z_chunk: np.ndarray) -> np.ndarray:
        """Decode one fixed-size latent chunk, advancing the cache.

        Args:
            z_chunk: ``(1, latent_dim, latent_frames)`` or equivalent
                shape; will be reshaped to canonical form.

        Returns:
            Decoded audio of shape ``(1, 1, samples_per_chunk)``.
        """
        expected = (1, self.latent_dim, self.latent_frames)
        z = np.ascontiguousarray(z_chunk, dtype=np.float32).reshape(expected)
        return self._predict_and_update_caches(z)

    def decode_batch(self, z_batch: np.ndarray) -> np.ndarray:
        """Decode a variable-length latent tensor, advancing the cache.

        For use with RangeDim models that accept variable ``latent_frames``.
        Accepts any ``(1, latent_dim, N)`` input where ``N`` is within
        the model's supported range.

        Args:
            z_batch: ``(1, latent_dim, N)`` latent features.

        Returns:
            Decoded audio of shape ``(1, 1, N * upsample_factor)``.
        """
        z = np.ascontiguousarray(z_batch, dtype=np.float32)
        if z.ndim == 2:
            z = z.reshape(1, z.shape[0], z.shape[1])
        if z.shape[0] != 1 or z.shape[1] != self.latent_dim:
            raise ValueError(f"expected shape (1, {self.latent_dim}, N), got {z.shape}")
        return self._predict_and_update_caches(z)

    def _predict_and_update_caches(self, z: np.ndarray) -> np.ndarray:
        """Run predict with cache I/O and update stored caches."""
        inputs = {"z": z}
        for i, c in enumerate(self._caches):
            inputs[f"cache_{i}"] = c

        result = self.model.predict(inputs)

        self._caches = [
            np.ascontiguousarray(result[f"new_cache_{i}"], dtype=np.float32)
            for i in range(len(self._caches))
        ]
        return result["audio"]
