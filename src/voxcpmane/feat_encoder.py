"""Numpy / CoreML inference wrapper for the fused VoxCPM2 feat_encoder.

Runtime counterpart to ``src/qeml/conversion/voxcpm2/feat_encoder.py``.

The converted ``.mlpackage`` is now channels-first NCHW:
``(chunk_patches, feat_dim, 1, patch_size) → (chunk_patches, lm_hidden, 1, 1)``.
The runtime presents the same outward interface as before — a
``(B, T, P, D) → (B, T, H_lm)`` function — by flattening + padding:

1. Rearrange ``(B, T, P, D) → (N, D, 1, P)`` where ``N = B·T``.
2. Pad ``N`` up to a multiple of ``chunk_patches`` (only the final
   chunk if needed).
3. Call the model ``⌈N / chunk_patches⌉`` times.
4. Drop the trailing singletons and reshape back to ``(B, T, H_lm)``.

Single-patch decode calls are fast: pad to one chunk and run once.
Prefill amortises across multiple chunks. Stateless — no cache
between calls.
"""

from __future__ import annotations

import math

from pathlib import Path

import coremltools as ct
import numpy as np

from ._coreml_utils import (
    get_feature_info,
    load_coreml_model,
)


class FeatEncoder:
    """Stateless wrapper around a converted fused feat_encoder ``.mlpackage``.

    The wrapped model's input is
    ``(chunk_patches, feat_dim, 1, patch_size)`` and its output is
    ``(chunk_patches, lm_hidden, 1, 1)``. This wrapper exposes
    :meth:`encode_patches`, which accepts the ``(B, T, P, D)`` layout
    used by ``VoxCPM2Model._inference`` and returns ``(B, T, H_lm)`` —
    a drop-in replacement for ``enc_to_lm_proj(feat_encoder(x))``.
    """

    def __init__(
        self,
        model_path: Path,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
    ):
        # ``ALL`` lets the scheduler pick between CPU, GPU and NE per op.
        # We deliberately do **not** default to ``CPU_ONLY``: with the
        # channels-first NCHW rewrite, a few ops (head-split matmul +
        # softmax + conv stack) lose significant precision on CoreML's
        # CPU fp16 path — cosine similarity collapsed from 0.9999 to
        # 0.41 in testing. Scheduling the same graph on GPU or the
        # Neural Engine keeps it at 0.9999+.
        self.model = load_coreml_model(model_path, compute_units=compute_units)

        info = get_feature_info(self.model, model_path, "patches")
        shape = info["shape"]
        enum_shapes = info["enumerated_shapes"]

        if len(shape) != 4:
            raise ValueError(
                f"expected 'patches' input of rank 4 (N, D, 1, P), got shape {shape}"
            )
        # NCHW: (chunk_patches, feat_dim, 1, patch_size)
        self.chunk_patches, self.feat_dim, _, self.patch_size = shape
        enum_chunk_patches = sorted(
            {
                enum_shape[0]
                for enum_shape in enum_shapes
                if len(enum_shape) == 4
                and enum_shape[1:] == (self.feat_dim, 1, self.patch_size)
            }
        )
        self.available_chunk_patches = tuple(enum_chunk_patches or [self.chunk_patches])
        self.prefill_chunk_patches = (
            16
            if 16 in self.available_chunk_patches
            else max(self.available_chunk_patches)
        )

    # ------------------------------------------------------------------ #
    # core API
    # ------------------------------------------------------------------ #
    def encode_patches(
        self,
        patches: np.ndarray,
        *,
        preferred_chunk_patches: int | None = None,
    ) -> np.ndarray:
        """Encode a ``(B, T, P, D)`` batch of patches to ``(B, T, H_lm)``.

        The CoreML model sees a flat ``(N, D, 1, P)`` batch; ``(B, T)``
        are recovered only at the output. The final chunk is zero-
        padded to ``chunk_patches`` and trimmed after concat.
        """
        B, T, P, D = patches.shape

        # (B, T, P, D) → (B*T, D, 1, P)
        flat = patches.reshape(B * T, P, D)
        flat = np.transpose(flat, (0, 2, 1))  # (N, D, P)
        flat = np.ascontiguousarray(flat[:, :, None, :], dtype=np.float32)
        out_flat = self._encode_flat(
            flat,
            chunk_patches=self._select_chunk_patches(preferred_chunk_patches),
        )  # (N, H_lm, 1, 1)
        return out_flat.reshape(B, T, -1)

    def _select_chunk_patches(self, preferred_chunk_patches: int | None) -> int:
        if preferred_chunk_patches is None:
            return int(self.chunk_patches)
        preferred = int(preferred_chunk_patches)
        if preferred in self.available_chunk_patches:
            return preferred
        candidates = [n for n in self.available_chunk_patches if n <= preferred]
        return int(max(candidates) if candidates else self.chunk_patches)

    def _encode_flat(self, patches: np.ndarray, *, chunk_patches: int) -> np.ndarray:
        """Run the CoreML model over an ``(N, D, 1, P)`` tensor.

        Pads ``N`` up to a multiple of the selected chunk size, calls the
        model ``⌈N / chunk_patches⌉`` times, concatenates, and trims
        the pad.
        """
        N = patches.shape[0]
        num_chunks = max(1, math.ceil(N / chunk_patches))
        padded_N = num_chunks * chunk_patches
        if padded_N != N:
            pad = np.zeros(
                (padded_N - N, self.feat_dim, 1, self.patch_size),
                dtype=np.float32,
            )
            patches = np.concatenate([patches, pad], axis=0)

        out_chunks = []
        for i in range(num_chunks):
            chunk = patches[i * chunk_patches : (i + 1) * chunk_patches]
            result = self.model.predict({"patches": chunk})
            out_chunks.append(result["lm_embed"])

        out = np.concatenate(out_chunks, axis=0)
        return out[:N]
