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

import json
from pathlib import Path
from typing import List, Tuple, Union

import coremltools as ct
import numpy as np

PathLike = Union[str, Path]

# Defaults matching AudioVAEConfig in upstream VoxCPM 2.
DEFAULT_LATENT_DIM = 64
DEFAULT_LATENT_FRAMES = 4
DEFAULT_UPSAMPLE_FACTOR = 1920  # math.prod([8, 6, 5, 2, 2, 2])
DEFAULT_PATCH_SIZE = 4
DEFAULT_OUT_SAMPLE_RATE = 48000


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


class AudioVAEDecoder:
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
        model_path: PathLike,
        latent_frames: int = DEFAULT_LATENT_FRAMES,
        latent_dim: int = DEFAULT_LATENT_DIM,
        upsample_factor: int = DEFAULT_UPSAMPLE_FACTOR,
        patch_size: int = DEFAULT_PATCH_SIZE,
        out_sample_rate: int = DEFAULT_OUT_SAMPLE_RATE,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_ONLY,
    ):
        self.model = _load_coreml_model(model_path, compute_units)
        self.latent_frames = int(latent_frames)
        self.latent_dim = int(latent_dim)
        self.upsample_factor = int(upsample_factor)
        self.patch_size = int(patch_size)
        self.out_sample_rate = int(out_sample_rate)

        self.samples_per_chunk = self.latent_frames * self.upsample_factor

        if Path(model_path).suffix == ".mlmodelc":
            metadata = _load_compiled_metadata_entry(model_path)
            self._cache_shapes = _discover_cache_shapes_from_schema(
                metadata.get("inputSchema", [])
            )
        else:
            self._cache_shapes = _discover_cache_shapes_from_spec_model(self.model)
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

        inputs = {"z": z}
        for i, c in enumerate(self._caches):
            inputs[f"cache_{i}"] = c

        result = self.model.predict(inputs)

        self._caches = [
            np.ascontiguousarray(result[f"new_cache_{i}"], dtype=np.float32)
            for i in range(len(self._caches))
        ]
        return result["audio"]

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
        z = np.transpose(patches, (2, 0, 1)).reshape(latent_dim, num_patches * patch_size)
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
            audio = self.decode_chunk(chunk)
            audio_chunks.append(audio.reshape(-1))
        waveform = np.concatenate(audio_chunks, axis=0)

        keep_samples = total_frames * self.upsample_factor
        return waveform[:keep_samples]


class AudioVAEDecoderStateful:
    """MLState variant of :class:`AudioVAEDecoder`.

    Same external surface — ``decode_chunk`` /
    :meth:`reset` / :meth:`decode_patches` — but the per-stage caches
    live inside the CoreML model as ``MLState`` buffers instead of
    crossing the Python ↔ CoreML boundary each step. The
    ``.mlpackage`` has only one input (``z``) and one output
    (``audio``); state is owned by a ``state`` handle carried on this
    object.
    """

    def __init__(
        self,
        model_path: PathLike,
        latent_frames: int = DEFAULT_LATENT_FRAMES,
        latent_dim: int = DEFAULT_LATENT_DIM,
        upsample_factor: int = DEFAULT_UPSAMPLE_FACTOR,
        patch_size: int = DEFAULT_PATCH_SIZE,
        out_sample_rate: int = DEFAULT_OUT_SAMPLE_RATE,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_ONLY,
    ):
        self.model = _load_coreml_model(model_path, compute_units)
        self.latent_frames = int(latent_frames)
        self.latent_dim = int(latent_dim)
        self.upsample_factor = int(upsample_factor)
        self.patch_size = int(patch_size)
        self.out_sample_rate = int(out_sample_rate)
        self.samples_per_chunk = self.latent_frames * self.upsample_factor

        self.state = self.model.make_state()

    def reset(self) -> None:
        """Re-create the state handle so every cache buffer is zeroed."""
        self.state = self.model.make_state()

    def decode_chunk(self, z_chunk: np.ndarray) -> np.ndarray:
        """Decode one fixed-size latent chunk, mutating state in place.

        Args:
            z_chunk: ``(1, latent_dim, latent_frames)`` or any shape that
                reshapes to it.

        Returns:
            Decoded audio of shape ``(1, 1, samples_per_chunk)``.
        """
        expected = (1, self.latent_dim, self.latent_frames)
        z = np.ascontiguousarray(z_chunk, dtype=np.float32).reshape(expected)
        result = self.model.predict({"z": z}, state=self.state)
        return result["audio"]

    def decode_patches(self, patches: np.ndarray) -> np.ndarray:
        """Drop-in mirror of :meth:`AudioVAEDecoder.decode_patches`."""
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

        z = np.transpose(patches, (2, 0, 1)).reshape(latent_dim, num_patches * patch_size)
        total_frames = z.shape[-1]

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


def _discover_cache_shapes_from_spec_model(model: ct.models.MLModel) -> List[Tuple[int, ...]]:
    """Read ``cache_0``, ``cache_1``, ... input shapes from the model spec."""
    spec_inputs = model.get_spec().description.input
    named: List[Tuple[int, Tuple[int, ...]]] = []
    for inp in spec_inputs:
        if not inp.name.startswith("cache_"):
            continue
        idx = int(inp.name[len("cache_"):])
        shape = tuple(int(d) for d in inp.type.multiArrayType.shape)
        named.append((idx, shape))
    named.sort(key=lambda kv: kv[0])
    return [shape for _, shape in named]


def _discover_cache_shapes_from_schema(input_schema: list[dict]) -> List[Tuple[int, ...]]:
    named: List[Tuple[int, Tuple[int, ...]]] = []
    for inp in input_schema:
        name = inp.get("name", "")
        if not name.startswith("cache_"):
            continue
        idx = int(name[len("cache_"):])
        shape = tuple(int(d) for d in inp["shape"].strip("[]").split(",") if d.strip())
        named.append((idx, shape))
    named.sort(key=lambda kv: kv[0])
    return [shape for _, shape in named]
