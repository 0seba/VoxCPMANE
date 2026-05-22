"""Shared CoreML loading and metadata utilities for VoxCPMANE2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np


def load_coreml_model(
    path: Path,
    *,
    compute_units: ct.ComputeUnit,
    function_name: Optional[str] = None,
):
    """Load a CoreML model from ``.mlpackage`` or compiled ``.mlmodelc``."""
    if is_compiled_model_path(path):
        return ct.models.CompiledMLModel(
            str(path),
            compute_units=compute_units,
            function_name=function_name,
        )
    return ct.models.MLModel(
        str(path),
        compute_units=compute_units,
        function_name=function_name,
    )


def is_compiled_model_path(path: Path) -> bool:
    """Return ``True`` when *path* points to a compiled ``.mlmodelc``."""
    return path.suffix == ".mlmodelc"


def resolve_model_path(
    path: Path,
    *,
    use_compiled: bool = True,
    compiled_dir: Optional[Path] = None,
) -> Path:
    """Resolve compiled counterpart (.mlmodelc) for a model package path."""
    compiled = path.with_suffix(".mlmodelc")
    if (
        use_compiled
        and compiled.exists()
        and (compiled / "metadata.json").exists()
        and (compiled / "model.mil").exists()
    ):
        return compiled
    if use_compiled and compiled_dir is not None:
        fallback = compiled_dir / path.with_suffix(".mlmodelc").name
        if (
            fallback.exists()
            and (fallback / "metadata.json").exists()
            and (fallback / "model.mil").exists()
        ):
            return fallback
    return path


def load_compiled_metadata_entry(path: Path) -> dict:
    """Read the first entry from the ``metadata.json`` sidecar."""
    metadata_path = path / "metadata.json"
    raw = json.loads(metadata_path.read_text())
    return raw[0]


def parse_shape_text(shape_text: str) -> Tuple[int, ...]:
    """Parse a ``\"[1, 64, 1, 4]\"``-style shape string into a tuple."""
    return tuple(int(d.strip()) for d in shape_text.strip("[]").split(",") if d.strip())


def _feature_info(
    shape,
    dtype,
    enumerated_shapes=(),
    shape_range=None,
) -> dict:
    return {
        "shape": tuple(shape),
        "dtype": np.dtype(dtype),
        "enumerated_shapes": tuple(enumerated_shapes),
        "shape_range": shape_range,
    }


def _compiled_dtype(name: str):
    return {
        "Float32": np.float32,
        "Float16": np.float16,
        "Int32": np.int32,
    }.get(name, np.float32)


def _load_json_shape_list(value: str | None, default):
    if value is None:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def discover_cache_shapes_from_spec_model(
    model: ct.models.MLModel,
) -> List[Tuple[int, ...]]:
    """Read ``cache_0``, ``cache_1``, … input shapes from the model spec."""
    spec_inputs = model.get_spec().description.input
    named: List[Tuple[int, Tuple[int, ...]]] = []
    for inp in spec_inputs:
        if not inp.name.startswith("cache_"):
            continue
        idx = int(inp.name[len("cache_") :])
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
        idx = int(name[len("cache_") :])
        shape = parse_shape_text(inp["shape"])
        named.append((idx, shape))
    named.sort(key=lambda kv: kv[0])
    return [shape for _, shape in named]


def discover_cache_shapes(
    model: ct.models.MLModel,
    path: Path,
) -> List[Tuple[int, ...]]:
    """Discover all cache input shapes from model spec or compiled metadata."""
    if is_compiled_model_path(path):
        metadata = load_compiled_metadata_entry(path)
        return discover_cache_shapes_from_schema(metadata.get("inputSchema", []))
    return discover_cache_shapes_from_spec_model(model)


def get_feature_info(
    model: ct.models.MLModel,
    path: Path,
    name: str,
    *,
    is_input: bool = True,
) -> dict:
    """Extract shape, dtype, enumerated shapes, and shape range for a feature."""
    from coremltools.proto import FeatureTypes_pb2

    if is_compiled_model_path(path):
        metadata = load_compiled_metadata_entry(path)
        schema_list = metadata.get("inputSchema" if is_input else "outputSchema", [])
        try:
            feature = next(f for f in schema_list if f.get("name") == name)
        except StopIteration:
            raise KeyError(f"Feature '{name}' not found in compiled metadata")

        shape = parse_shape_text(feature.get("shape", ""))
        enum_shapes = _load_json_shape_list(feature.get("enumeratedShapes"), ())
        shape_range = _load_json_shape_list(feature.get("shapeRange"), None)
        return _feature_info(
            shape,
            _compiled_dtype(feature.get("dataType", "")),
            (tuple(int(d) for d in s) for s in enum_shapes),
            [(int(r[0]), int(r[1])) for r in shape_range] if shape_range else None,
        )
    else:
        spec = model.get_spec()
        functions = getattr(spec.description, "functions", [])
        if functions:
            target = spec.description.defaultFunctionName or "main"
            func_desc = next(
                (f for f in functions if f.name == target), spec.description
            )
        else:
            func_desc = spec.description

        schema_list = func_desc.input if is_input else func_desc.output
        try:
            feature = next(f for f in schema_list if f.name == name)
        except StopIteration:
            raise KeyError(f"Feature '{name}' not found in spec")

        multi_array_type = feature.type.multiArrayType
        shape = tuple(int(d) for d in multi_array_type.shape)
        enumerated = getattr(multi_array_type, "enumeratedShapes", None)
        shape_range_msg = getattr(multi_array_type, "shapeRange", None)
        dtype_by_spec = {
            FeatureTypes_pb2.ArrayFeatureType.FLOAT32: np.float32,
            FeatureTypes_pb2.ArrayFeatureType.FLOAT16: np.float16,
            FeatureTypes_pb2.ArrayFeatureType.INT32: np.int32,
        }
        size_ranges = (
            getattr(shape_range_msg, "sizeRanges", None)
            if shape_range_msg is not None
            else None
        )
        return _feature_info(
            shape,
            dtype_by_spec.get(multi_array_type.dataType, np.float32),
            (
                tuple(int(dim) for dim in shape_msg.shape)
                for shape_msg in getattr(enumerated, "shapes", [])
            ),
            (
                [(int(r.lowerBound), int(r.upperBound)) for r in size_ranges]
                if size_ranges
                else None
            ),
        )

