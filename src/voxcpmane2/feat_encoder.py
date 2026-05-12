"""Numpy / CoreML inference wrapper for the fused VoxCPM2 feat_encoder.

Runtime counterpart to ``src/qeml/conversion/voxcpm2/feat_encoder.py``.
No PyTorch — only numpy and coremltools.

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
import json
from pathlib import Path
from typing import Union

import coremltools as ct
import numpy as np

PathLike = Union[str, Path]


def _load_coreml_model(path: PathLike, compute_units: ct.ComputeUnit):
    model_path = Path(path)
    if model_path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(model_path), compute_units=compute_units)
    return ct.models.MLModel(str(model_path), compute_units=compute_units)


def _load_compiled_metadata_entry(path: PathLike) -> dict:
    model_path = Path(path)
    metadata_path = model_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing compiled metadata file: {metadata_path}")
    raw = json.loads(metadata_path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"unexpected metadata format in {metadata_path}")
    return raw[0]


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
        model_path: PathLike,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
    ):
        # ``ALL`` lets the scheduler pick between CPU, GPU and NE per op.
        # We deliberately do **not** default to ``CPU_ONLY``: with the
        # channels-first NCHW rewrite, a few ops (head-split matmul +
        # softmax + conv stack) lose significant precision on CoreML's
        # CPU fp16 path — cosine similarity collapsed from 0.9999 to
        # 0.41 in testing. Scheduling the same graph on GPU or the
        # Neural Engine keeps it at 0.9999+.
        self.model = _load_coreml_model(model_path, compute_units)
        if Path(model_path).suffix == ".mlmodelc":
            metadata = _load_compiled_metadata_entry(model_path)
            input_schema = metadata.get("inputSchema", [])
            patches_input = next(i for i in input_schema if i.get("name") == "patches")
            shape = tuple(int(d) for d in patches_input["shape"].strip("[]").split(",") if d.strip())
        else:
            spec_inputs = self.model.get_spec().description.input
            patches_input = next(i for i in spec_inputs if i.name == "patches")
            shape = tuple(int(d) for d in patches_input.type.multiArrayType.shape)
        if len(shape) != 4:
            raise ValueError(
                f"expected 'patches' input of rank 4 (N, D, 1, P), got shape {shape}"
            )
        # NCHW: (chunk_patches, feat_dim, 1, patch_size)
        self.chunk_patches, self.feat_dim, _, self.patch_size = shape

    # ------------------------------------------------------------------ #
    # core API
    # ------------------------------------------------------------------ #
    def encode_patches(self, patches: np.ndarray) -> np.ndarray:
        """Encode a ``(B, T, P, D)`` batch of patches to ``(B, T, H_lm)``.

        The CoreML model sees a flat ``(N, D, 1, P)`` batch; ``(B, T)``
        are recovered only at the output. The final chunk is zero-
        padded to ``chunk_patches`` and trimmed after concat.
        """
        if patches.ndim != 4:
            raise ValueError(
                f"expected patches of shape (B, T, P, D), got {patches.shape}"
            )
        B, T, P, D = patches.shape
        if P != self.patch_size or D != self.feat_dim:
            raise ValueError(
                f"patch_size / feat_dim mismatch: model expects "
                f"({self.patch_size}, {self.feat_dim}), got ({P}, {D})"
            )

        # (B, T, P, D) → (B*T, D, 1, P)
        flat = patches.reshape(B * T, P, D)
        flat = np.transpose(flat, (0, 2, 1))  # (N, D, P)
        flat = np.ascontiguousarray(flat[:, :, None, :], dtype=np.float32)
        out_flat = self._encode_flat(flat)  # (N, H_lm, 1, 1)
        return out_flat.reshape(B, T, -1)

    def _encode_flat(self, patches: np.ndarray) -> np.ndarray:
        """Run the CoreML model over an ``(N, D, 1, P)`` tensor.

        Pads ``N`` up to a multiple of ``chunk_patches``, calls the
        model ``⌈N / chunk_patches⌉`` times, concatenates, and trims
        the pad.
        """
        N = patches.shape[0]
        num_chunks = max(1, math.ceil(N / self.chunk_patches))
        padded_N = num_chunks * self.chunk_patches
        if padded_N != N:
            pad = np.zeros(
                (padded_N - N, self.feat_dim, 1, self.patch_size),
                dtype=np.float32,
            )
            patches = np.concatenate([patches, pad], axis=0)

        out_chunks = []
        for i in range(num_chunks):
            chunk = patches[i * self.chunk_patches : (i + 1) * self.chunk_patches]
            result = self.model.predict({"patches": chunk})
            out_chunks.append(result["lm_embed"])

        out = np.concatenate(out_chunks, axis=0)
        return out[:N]
