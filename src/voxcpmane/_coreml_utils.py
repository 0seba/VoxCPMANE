"""Shared CoreML loading and metadata utilities for VoxCPMANE2.

Centralises the model-loading, metadata-reading, and cache-shape-discovery
helpers that were previously duplicated across ``audio_vae_encoder``,
``audio_vae_decoder``, ``feat_encoder``, ``lm``, ``locdit``, and
``generator``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import coremltools as ct

PathLike = Union[str, Path]


# ------------------------------------------------------------------ #
# Model loading
# ------------------------------------------------------------------ #

def load_coreml_model(
    path: PathLike,
    *,
    compute_units: ct.ComputeUnit,
    function_name: Optional[str] = None,
):
    """Load a CoreML model from ``.mlpackage`` or compiled ``.mlmodelc``.

    Args:
        path: Path to the model directory.
        compute_units: CoreML compute-unit selection.
        function_name: Optional multifunction entry-point name (only
            supported for ``MLModel`` / ``CompiledMLModel`` constructors
            that accept it).
    """
    model_path = Path(path)
    if model_path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(
            str(model_path),
            compute_units=compute_units,
            function_name=function_name,
        )
    return ct.models.MLModel(
        str(model_path),
        compute_units=compute_units,
        function_name=function_name,
    )


def is_compiled_model_path(path: PathLike) -> bool:
    """Return ``True`` when *path* points to a compiled ``.mlmodelc``."""
    return Path(path).suffix == ".mlmodelc"


# ------------------------------------------------------------------ #
# Metadata
# ------------------------------------------------------------------ #

def load_compiled_metadata_entry(path: PathLike) -> dict:
    """Read the first entry from the ``metadata.json`` sidecar.

    Compiled ``.mlmodelc`` directories produced by our conversion pipeline
    carry a ``metadata.json`` list; this helper returns the first element.
    """
    model_path = Path(path)
    metadata_path = model_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing compiled metadata file: {metadata_path}")
    raw = json.loads(metadata_path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"unexpected metadata format in {metadata_path}")
    return raw[0]


def parse_shape_text(shape_text: str) -> Tuple[int, ...]:
    """Parse a ``\"[1, 64, 1, 4]\"``-style shape string into a tuple."""
    return tuple(int(d.strip()) for d in shape_text.strip("[]").split(",") if d.strip())


# ------------------------------------------------------------------ #
# Cache-shape discovery (AudioVAE encoder / decoder)
# ------------------------------------------------------------------ #

def discover_cache_shapes_from_spec_model(
    model: ct.models.MLModel,
) -> List[Tuple[int, ...]]:
    """Read ``cache_0``, ``cache_1``, … input shapes from the model spec."""
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


def discover_cache_shapes_from_schema(
    input_schema: list[dict],
) -> List[Tuple[int, ...]]:
    """Read ``cache_0``, ``cache_1``, … shapes from a compiled metadata schema."""
    named: List[Tuple[int, Tuple[int, ...]]] = []
    for inp in input_schema:
        name = inp.get("name", "")
        if not name.startswith("cache_"):
            continue
        idx = int(name[len("cache_"):])
        shape = parse_shape_text(inp["shape"])
        named.append((idx, shape))
    named.sort(key=lambda kv: kv[0])
    return [shape for _, shape in named]


# ------------------------------------------------------------------ #
# Chunk-size selection (shared by LM wrappers)
# ------------------------------------------------------------------ #

def select_chunk_size(
    remaining: int,
    function_names_by_chunk_size: dict[int, str],
    default_chunk_size: int,
    *,
    preferred_chunk_size: int | None = None,
) -> int:
    """Pick the best chunk size for *remaining* tokens.

    If *preferred_chunk_size* is given and available, return it directly.
    Otherwise, return the largest available size ≤ *remaining*, falling
    back to *default_chunk_size*.
    """
    if preferred_chunk_size is not None:
        preferred = int(preferred_chunk_size)
        if preferred not in function_names_by_chunk_size:
            raise ValueError(
                f"preferred chunk size {preferred} is not available; "
                f"available sizes are {sorted(function_names_by_chunk_size)}"
            )
        return preferred
    for chunk_size in sorted(function_names_by_chunk_size, reverse=True):
        if chunk_size <= remaining:
            return chunk_size
    return default_chunk_size
