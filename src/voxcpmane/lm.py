"""Runtime wrapper for stateful VoxCPM2 CoreML language models.

The converted model has a fixed CoreML input shape
``(1, channels, 1, chunk_size)`` and persistent MLState KV caches.  This
wrapper presents the subset of ``MiniCPMModel`` used by
``VoxCPM2Generator``:

* ``forward(inputs_embeds, is_causal=True) -> (hidden_states, None)``
* ``forward_step(inputs_embeds, position_id) -> hidden_states``
* ``kv_cache.fill_caches(...)`` and ``kv_cache.step()``
"""

from __future__ import annotations

import time
import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import coremltools as ct
import numpy as np

from ._coreml_utils import (
    is_compiled_model_path,
    load_compiled_metadata_entry,
    load_coreml_model,
    parse_shape_text,
)


class _RuntimeKVCache:
    """Compatibility shim for VoxCPM2's ``lm.kv_cache`` calls."""

    def __init__(self, owner: "CoreMLMiniCPMLM"):
        self._owner = owner

    def fill_caches(self, *_args: Any, **_kwargs: Any) -> None:
        # CoreML MLState was already populated by the prefill calls.
        return None

    def step(self) -> int:
        # Upstream uses this value only to pass a position tensor into
        # ``forward_step``.  The wrapper tracks/advances the position
        # internally, so this is an observation, not a mutation.
        return self._owner.current_position


class CoreMLMiniCPMLM:
    """Stateful wrapper around a converted MiniCPM LM ``.mlpackage``."""

    def __init__(
        self,
        model_path: Path | Sequence[Path],
        *,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE,
        embed_tokens: Optional[Any] = None,
        eager_load_functions: bool = False,
        max_function_handles: int = 1,
        unload_inactive_functions: bool = False,
        idle_prefill_chunk_size: Optional[int] = None,
        preload_chunk_sizes: Optional[List[int]] = None,
        keep_default_function_loaded: bool = False,
        restrict_to_preload: bool = False,
        startup_decode_warmup_repeats: int = 0,
        label: str = "lm",
    ):
        self.compute_units = compute_units
        self.embed_tokens = embed_tokens
        self.kv_cache = _RuntimeKVCache(self)
        self.current_position = 0
        self.label = label

        # Support sequence of paths/models (chained model)
        if isinstance(model_path, (list, tuple, Sequence)) and not isinstance(
            model_path, (str, Path)
        ):
            self.is_chain = True
            model_paths = list(model_path)
            if len(model_paths) < 1:
                raise ValueError("at least one model path required")
            self.submodels = [
                CoreMLMiniCPMLM(
                    p,
                    compute_units=compute_units,
                    embed_tokens=embed_tokens,
                    eager_load_functions=eager_load_functions,
                    max_function_handles=max_function_handles,
                    unload_inactive_functions=unload_inactive_functions,
                    idle_prefill_chunk_size=idle_prefill_chunk_size,
                    preload_chunk_sizes=preload_chunk_sizes,
                    keep_default_function_loaded=keep_default_function_loaded,
                    restrict_to_preload=restrict_to_preload,
                    startup_decode_warmup_repeats=startup_decode_warmup_repeats,
                    label=f"{label}/part{i}",
                )
                for i, p in enumerate(model_paths)
            ]
            
            shapes = [
                (m.batch_size, m.input_channels, m.spatial_dim, m.chunk_size)
                for m in self.submodels
            ]
            if not all(
                s[0] == shapes[0][0] and s[2] == shapes[0][2] and s[3] == shapes[0][3]
                for s in shapes
            ):
                raise ValueError(f"submodel input shapes differ in B/H/S: {shapes}")
            chunk_sets = [set(m._function_names_by_chunk_size) for m in self.submodels]
            if not all(chunks == chunk_sets[0] for chunks in chunk_sets):
                raise ValueError(f"submodel multifunction lengths differ: {chunk_sets}")

            self.batch_size = shapes[0][0]
            self.input_channels = shapes[0][1]
            self.spatial_dim = shapes[0][2]
            self.chunk_size = shapes[0][3]
            self.cache_length = min(
                (m.cache_length for m in self.submodels if m.cache_length is not None),
                default=None,
            )
            self._function_names_by_chunk_size = dict(
                self.submodels[0]._function_names_by_chunk_size
            )
            return

        self.is_chain = False
        self.model_path = model_path
        spec = None
        metadata = None
        if is_compiled_model_path(self.model_path):
            metadata = load_compiled_metadata_entry(self.model_path)
            default_function_name = "main"
        else:
            spec_model = ct.models.MLModel(str(self.model_path), skip_model_load=True)
            spec = spec_model.get_spec()
            default_function_name = spec.description.defaultFunctionName or "main"
        self.model = None
        self.state: Optional[ct.models.model.MLState] = None
        self.eager_load_functions = bool(eager_load_functions)
        self.max_function_handles = int(max_function_handles)
        self.unload_inactive_functions = bool(unload_inactive_functions)
        self.keep_default_function_loaded = bool(keep_default_function_loaded)
        self.lifecycle_events: list[tuple[str, float]] = []
        self.idle_prefill_chunk_size = (
            int(idle_prefill_chunk_size)
            if idle_prefill_chunk_size is not None
            else None
        )

        if metadata is not None:
            fn_desc = self._metadata_function_for_name(metadata, default_function_name)
            input_schema = fn_desc.get("inputSchema", [])
            x_desc = next(
                (i for i in input_schema if i.get("name") == "inputs_embeds"), None
            )
            if x_desc is None:
                raise ValueError("model has no 'inputs_embeds' input")
            position_desc = next(
                (i for i in input_schema if i.get("name") == "position_id"), None
            )
            if position_desc is None:
                raise ValueError("model has no 'position_id' input")
            seed_desc = next(
                (i for i in input_schema if i.get("name") == "position_index_seed"),
                None,
            )
            shape = parse_shape_text(x_desc["shape"])
            self.position_id_shape = parse_shape_text(position_desc["shape"])
            self.position_index_seed_shape = (
                parse_shape_text(seed_desc["shape"]) if seed_desc is not None else None
            )
            state_shapes = self._metadata_state_shapes(
                fn_desc
            ) or self._metadata_state_shapes(metadata)
        else:
            model_desc = self._description_for_function(spec, default_function_name)
            input_desc = model_desc.input
            try:
                x_desc = next(i for i in input_desc if i.name == "inputs_embeds")
            except StopIteration as exc:
                raise ValueError("model has no 'inputs_embeds' input") from exc
            try:
                position_desc = next(i for i in input_desc if i.name == "position_id")
            except StopIteration as exc:
                raise ValueError("model has no 'position_id' input") from exc
            seed_desc = next(
                (i for i in input_desc if i.name == "position_index_seed"),
                None,
            )
            shape = tuple(int(d) for d in x_desc.type.multiArrayType.shape)
            self.position_id_shape = tuple(
                int(d) for d in position_desc.type.multiArrayType.shape
            )
            self.position_index_seed_shape = (
                tuple(int(d) for d in seed_desc.type.multiArrayType.shape)
                if seed_desc is not None
                else None
            )
            state_shapes = self._spec_state_shapes(spec)

        self.state_shapes = list(state_shapes)
        self.state_shape_by_name = {name: tuple(shape) for name, shape in state_shapes}

        if len(shape) != 4:
            raise ValueError(
                f"expected rank-4 'inputs_embeds' shape (1, C, 1, S), got {shape}"
            )
        self.batch_size, self.input_channels, self.spatial_dim, self.chunk_size = shape
        if self.batch_size != 1 or self.spatial_dim != 1:
            raise ValueError(
                "only single-batch rank-4 LM models are supported for now; "
                f"got shape {shape}"
            )
        self.cache_length = self._cache_length_from_state_shapes(state_shapes)
        self._models_by_chunk_size = {}
        self._function_names_by_chunk_size = {self.chunk_size: default_function_name}
        if metadata is not None:
            self._discover_multifunction_functions_from_metadata(metadata)
        else:
            self._discover_multifunction_functions(spec)
        decode_warmup_repeats = max(0, int(startup_decode_warmup_repeats))
        decode_chunk_size = self.chunk_size
        if decode_chunk_size not in self._function_names_by_chunk_size:
            decode_chunk_size = min(self._function_names_by_chunk_size)
        if preload_chunk_sizes:
            self.model = None
            self.max_function_handles = max(
                self.max_function_handles, len(preload_chunk_sizes)
            )
            if restrict_to_preload:
                self._function_names_by_chunk_size = {
                    cs: name
                    for cs, name in self._function_names_by_chunk_size.items()
                    if cs in preload_chunk_sizes
                }
            if decode_warmup_repeats > 0:
                prefill_warmup_sizes = [
                    cs for cs in preload_chunk_sizes if cs != decode_chunk_size
                ]
                for cs in prefill_warmup_sizes:
                    if cs not in self._function_names_by_chunk_size:
                        raise ValueError(
                            f"preload chunk size {cs} not available in model; "
                            f"available: {sorted(self._function_names_by_chunk_size)}"
                        )
                    self.warmup_function(
                        cs,
                        repeats=decode_warmup_repeats,
                        profile_prefix="startup_prefill_warmup",
                    )
                    self._unload_function(
                        cs,
                        event_name=f"startup_prefill_warmup/unload_function_s{cs}",
                    )
                if prefill_warmup_sizes:
                    gc.collect()
                self.warmup_function(
                    decode_chunk_size,
                    repeats=decode_warmup_repeats,
                    profile_prefix="startup_decode_warmup",
                )
                self._unload_function(
                    decode_chunk_size,
                    event_name=f"startup_decode_warmup/unload_function_s{decode_chunk_size}",
                )
                if decode_chunk_size == self.chunk_size:
                    self.model = None
                gc.collect()
            for cs in preload_chunk_sizes:
                if cs in self._function_names_by_chunk_size:
                    self._model_for_chunk_size(cs, profile_prefix="preload")
                else:
                    raise ValueError(
                        f"preload chunk size {cs} not available in model; "
                        f"available: {sorted(self._function_names_by_chunk_size)}"
                    )
        else:
            if decode_warmup_repeats > 0:
                self.warmup_function(
                    decode_chunk_size,
                    repeats=decode_warmup_repeats,
                    profile_prefix="startup_decode_warmup",
                )
                self._unload_function(
                    decode_chunk_size,
                    event_name=f"startup_decode_warmup/unload_function_s{decode_chunk_size}",
                )
                if decode_chunk_size == self.chunk_size:
                    self.model = None
                gc.collect()
            self.model = self._load_default_model_func(default_function_name)
            self._models_by_chunk_size[self.chunk_size] = self.model
            if (
                self.unload_inactive_functions
                and self.idle_prefill_chunk_size is not None
            ):
                if (
                    self.idle_prefill_chunk_size
                    not in self._function_names_by_chunk_size
                ):
                    raise ValueError(
                        f"idle prefill chunk size {self.idle_prefill_chunk_size} "
                        f"is not available; available sizes are "
                        f"{sorted(self._function_names_by_chunk_size)}"
                    )
                self._unload_function(
                    self.chunk_size,
                    event_name=f"init/unload_function_s{self.chunk_size}",
                )
                self.model = None
                self._model_for_chunk_size(
                    self.idle_prefill_chunk_size,
                    profile_prefix="init",
                )
            if self.eager_load_functions:
                n_functions = len(self._function_names_by_chunk_size)
                if self.max_function_handles < n_functions:
                    self.max_function_handles = n_functions
                self._load_multifunction_handles()
        self._ensure_persistent_state()

    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[Any, None]:
        return self.forward(*args, **kwargs)

    def reset(self) -> None:
        if self.is_chain:
            for submodel in self.submodels:
                submodel.reset()
            self.current_position = 0
            return
        model = self._active_or_default_model()
        self._ensure_persistent_state(model)
        self.current_position = 0

    def snapshot_state_prefix(self, prefix_length: int) -> dict[str, np.ndarray]:
        """Read populated KV prefix state as float16 arrays in a flat dict."""
        if self.is_chain:
            flat = {}
            for idx, submodel in enumerate(self.submodels):
                sub_snapshot = submodel.snapshot_state_prefix(prefix_length)
                for name, arr in sub_snapshot.items():
                    flat[f"part_{idx}:{name}"] = arr
            return flat

        if self.state is None:
            raise RuntimeError("cannot snapshot LM state before it has been created")
        prefix_length = int(prefix_length)
        arrays = {}
        for name, shape in self.state_shapes:
            value = np.asarray(self.state.read_state(name))
            if value.ndim >= 3 and value.shape[2] >= prefix_length:
                index = [slice(None)] * value.ndim
                index[2] = slice(0, prefix_length)
                value = value[tuple(index)]
            arrays[name] = value.astype(np.float16, copy=False)
        return arrays

    def restore_state_prefix(self, snapshot: dict[str, np.ndarray], position: int) -> None:
        """Restore a float16 KV prefix snapshot into the persistent MLState."""
        if self.is_chain:
            for idx, submodel in enumerate(self.submodels):
                sub_snapshot = {
                    name.split(":", 1)[1]: arr
                    for name, arr in snapshot.items()
                    if name.startswith(f"part_{idx}:")
                }
                submodel.restore_state_prefix(sub_snapshot, position)
            self.current_position = position
            return

        model = self._active_or_default_model()
        self._ensure_persistent_state(model)
        for name, cached in snapshot.items():
            shape = self.state_shape_by_name.get(name)
            if shape is None:
                raise ValueError(f"state {name!r} is not present in this LM")
            full = np.zeros(shape, dtype=np.float32)
            cached_arr = np.asarray(cached, dtype=np.float32)
            if cached_arr.ndim >= 3 and full.ndim >= 3:
                index = [slice(None)] * full.ndim
                index[2] = slice(0, cached_arr.shape[2])
                full[tuple(index)] = cached_arr
            else:
                full[...] = cached_arr
            self.state.write_state(name, full)
        self.current_position = position

    def forward(
        self,
        inputs_embeds: np.ndarray,
        is_causal: bool = True,
        reset_state: bool = True,
        profiler: Optional[Any] = None,
        profile_prefix: str = "lm",
        preferred_chunk_size: Optional[int] = None,
    ) -> Tuple[Any, None]:
        """Run prefill over ``inputs_embeds`` and return ``(hidden, None)``."""
        _ = is_causal  # CoreML graph is always causal.
        if self.is_chain and profile_prefix == "lm":
            profile_prefix = "lm_chain"
        output = self._run(
            inputs_embeds,
            reset_state=reset_state,
            profiler=profiler,
            profile_prefix=profile_prefix,
            preferred_chunk_size=preferred_chunk_size,
        )
        return output, None

    def forward_step(
        self,
        inputs_embeds: np.ndarray,
        position_id: Any = None,
        profiler: Optional[Any] = None,
        profile_prefix: str = "lm",
    ) -> Any:
        """Run one decode step or a small padded decode chunk."""
        _ = position_id  # Position is tracked by ``current_position``.
        if self.is_chain and profile_prefix == "lm":
            profile_prefix = "lm_chain"
        return self._run(
            inputs_embeds,
            reset_state=False,
            profiler=profiler,
            profile_prefix=profile_prefix,
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _run(
        self,
        inputs_embeds: np.ndarray,
        *,
        reset_state: bool,
        profiler: Optional[Any] = None,
        profile_prefix: str = "lm",
        preferred_chunk_size: Optional[int] = None,
    ) -> Any:
        if self.is_chain:
            with self._profile(profiler, f"{profile_prefix}/to_numpy_nchw"):
                arr, original_rank = self._to_numpy(inputs_embeds)
                x = self._to_nchw(arr)

            if x.shape[0] != 1:
                raise ValueError(f"CoreML LM chain supports B=1, got B={x.shape[0]}")
            if x.shape[1] != self.input_channels:
                raise ValueError(
                    f"chain expects {self.input_channels} input channels, got {x.shape[1]}"
                )

            length = x.shape[-1]
            chunks = []
            offset = 0
            chunk_size = self._select_runtime_chunk_size(preferred_chunk_size)
            while offset < length:
                remaining = length - offset
                consumed = min(remaining, chunk_size)
                with self._profile(
                    profiler, f"{profile_prefix}/chunk_prepare_s{chunk_size}"
                ):
                    chunk = np.ascontiguousarray(
                        x[..., offset : offset + consumed],
                        dtype=np.float32,
                    )
                    hidden = chunk
                for idx, submodel in enumerate(self.submodels):
                    hidden = submodel._run(
                        hidden,
                        reset_state=reset_state,
                        profiler=profiler,
                        profile_prefix=f"{profile_prefix}/part{idx}",
                        preferred_chunk_size=chunk_size,
                    )
                reset_state = False
                chunks.append(hidden)
                self.current_position += consumed
                offset += consumed

            with self._profile(profiler, f"{profile_prefix}/concat_restore"):
                out = np.concatenate(chunks, axis=-1)[..., :length]
                return self._from_nchw(out, original_rank)

        # Single model execution
        with self._profile(profiler, f"{profile_prefix}/to_numpy_nchw"):
            arr, original_rank = self._to_numpy(inputs_embeds)
            x = self._to_nchw(arr)

        if x.shape[0] != 1:
            raise ValueError(f"CoreML LM state path supports B=1, got B={x.shape[0]}")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"model expects {self.input_channels} input channels, got {x.shape[1]}"
            )

        length = x.shape[-1]
        chunks = []
        offset = 0
        state_needs_reset = bool(reset_state)
        if state_needs_reset:
            self.current_position = 0
        chunk_size = self._select_runtime_chunk_size(preferred_chunk_size)
        while offset < length:
            remaining = length - offset
            with self._profile(
                profiler, f"{profile_prefix}/chunk_prepare_s{chunk_size}"
            ):
                chunk = np.ascontiguousarray(
                    x[..., offset : offset + chunk_size],
                    dtype=np.float32,
                )
                pad = chunk_size - chunk.shape[-1]
                if pad:
                    chunk = np.pad(chunk, ((0, 0), (0, 0), (0, 0), (0, pad)))
                position_id = self._position_id_value()
                model_inputs = {"inputs_embeds": chunk, "position_id": position_id}
                if self.position_index_seed_shape is not None:
                    model_inputs["position_index_seed"] = np.zeros(
                        (chunk_size,), dtype=np.int32
                    )
            model = self._model_for_chunk_size(
                chunk_size,
                profiler=profiler,
                profile_prefix=profile_prefix,
            )
            if self.state is None:
                with self._profile(
                    profiler, f"{profile_prefix}/make_state_s{chunk_size}"
                ):
                    self._ensure_persistent_state(model)
            state_needs_reset = False
            with self._profile(profiler, f"{profile_prefix}/predict_s{chunk_size}"):
                result = model.predict(model_inputs, state=self.state)
                hidden_states = result["hidden_states"]
            chunks.append(hidden_states[..., :remaining] if pad else hidden_states)
            self.current_position += remaining if pad else chunk_size
            offset += remaining if pad else chunk_size

        with self._profile(profiler, f"{profile_prefix}/concat_restore"):
            out = np.concatenate(chunks, axis=-1)
            return self._from_nchw(out, original_rank)

    def _record_multifunction_function(
        self,
        name: str,
        shape: tuple[int, ...],
        seed_shape: tuple[int, ...] | None,
    ) -> None:
        if not name.startswith("length_"):
            return
        try:
            chunk_size = int(name.removeprefix("length_"))
        except ValueError:
            return
        if chunk_size in self._models_by_chunk_size:
            return
        expected = (self.batch_size, self.input_channels, self.spatial_dim, chunk_size)
        if tuple(shape) != expected:
            raise ValueError(
                f"function {name!r} has incompatible inputs_embeds shape {shape}; "
                f"expected {expected}"
            )
        if seed_shape is not None and tuple(seed_shape) != (chunk_size,):
            raise ValueError(
                f"function {name!r} has incompatible position_index_seed "
                f"shape {seed_shape}, expected {(chunk_size,)}"
            )
        self._function_names_by_chunk_size[chunk_size] = name

    def _discover_multifunction_functions(self, spec: Any) -> None:
        """Record ``length_<N>`` functions from a multifunction package."""
        for function in spec.description.functions:
            x_desc = next((i for i in function.input if i.name == "inputs_embeds"), None)
            if x_desc is None:
                continue
            seed_desc = next(
                (i for i in function.input if i.name == "position_index_seed"),
                None,
            )
            self._record_multifunction_function(
                function.name,
                tuple(int(d) for d in x_desc.type.multiArrayType.shape),
                (
                    tuple(int(d) for d in seed_desc.type.multiArrayType.shape)
                    if seed_desc is not None
                    else None
                ),
            )

    def _discover_multifunction_functions_from_metadata(self, metadata: dict) -> None:
        for function in metadata.get("functions", []):
            name = function.get("name", "")
            input_schema = function.get("inputSchema", [])
            x_desc = next(
                (i for i in input_schema if i.get("name") == "inputs_embeds"), None
            )
            if x_desc is None:
                continue
            seed_desc = next(
                (i for i in input_schema if i.get("name") == "position_index_seed"),
                None,
            )
            self._record_multifunction_function(
                name,
                parse_shape_text(x_desc["shape"]),
                parse_shape_text(seed_desc["shape"]) if seed_desc is not None else None,
            )

    def _load_multifunction_handles(self) -> None:
        """Eagerly load all discovered ``length_<N>`` functions."""
        for chunk_size in sorted(self._function_names_by_chunk_size):
            self._model_for_chunk_size(chunk_size)

    def _select_runtime_chunk_size(self, preferred_chunk_size: Optional[int]) -> int:
        if preferred_chunk_size is not None:
            preferred = int(preferred_chunk_size)
            if preferred in self._function_names_by_chunk_size:
                return preferred
        if self.chunk_size in self._function_names_by_chunk_size:
            return self.chunk_size
        return min(self._function_names_by_chunk_size)

    def warmup_loaded_functions(
        self,
        *,
        repeats: int = 5,
        profile_prefix: str = "startup_warmup",
    ) -> None:
        """Run synthetic single predicts through every currently loaded handle."""
        if self.is_chain:
            for idx, submodel in enumerate(self.submodels):
                submodel.warmup_loaded_functions(
                    repeats=repeats,
                    profile_prefix=f"{profile_prefix}/part{idx}",
                )
            self.current_position = 0
            return
        loaded = [
            (chunk_size, self._models_by_chunk_size[chunk_size])
            for chunk_size in sorted(self._models_by_chunk_size)
        ]
        for chunk_size, model in loaded:
            self._warmup_model(
                model,
                chunk_size,
                repeats=repeats,
            )

    def warmup_function(
        self,
        chunk_size: int,
        *,
        repeats: int = 5,
        profile_prefix: str = "startup_warmup",
    ) -> None:
        """Load one function handle and execute independent dummy predicts."""
        if self.is_chain:
            for idx, submodel in enumerate(self.submodels):
                submodel.warmup_function(
                    chunk_size,
                    repeats=repeats,
                    profile_prefix=f"{profile_prefix}/part{idx}",
                )
            self.current_position = 0
            return

        repeats = max(0, int(repeats))
        if repeats == 0:
            return
        t0 = time.perf_counter()
        model = self._model_for_chunk_size(chunk_size, profile_prefix=profile_prefix)
        self._warmup_model(model, chunk_size, repeats=repeats)
        if profile_prefix.startswith("startup"):
            print(
                f"   ✓ {self.label} {profile_prefix} length_{chunk_size}: "
                f"{time.perf_counter() - t0:.3f}s",
                flush=True,
            )

    def _warmup_model(
        self,
        model: ct.models.MLModel,
        chunk_size: int,
        *,
        repeats: int,
    ) -> None:
        x = np.zeros(
            (self.batch_size, self.input_channels, self.spatial_dim, int(chunk_size)),
            dtype=np.float32,
        )
        for _ in range(repeats):
            self.current_position = 0
            state = model.make_state()
            model_inputs = {
                "inputs_embeds": x,
                "position_id": self._position_id_value(),
            }
            if self.position_index_seed_shape is not None:
                model_inputs["position_index_seed"] = np.zeros(
                    (int(chunk_size),), dtype=np.int32
                )
            model.predict(model_inputs, state=state)
        self.state = None
        self.current_position = 0

    def _active_or_default_model(self) -> Any:
        model = self._models_by_chunk_size.get(self.chunk_size)
        if model is not None:
            return model
        if self._models_by_chunk_size:
            return next(iter(self._models_by_chunk_size.values()))
        cs = self.chunk_size
        if cs not in self._function_names_by_chunk_size:
            cs = min(self._function_names_by_chunk_size)
        return self._model_for_chunk_size(cs)

    def _load_default_model_func(self, default_function_name: str) -> Any:
        t0 = time.perf_counter()
        try:
            return load_coreml_model(
                self.model_path,
                compute_units=self.compute_units,
                function_name=default_function_name,
            )
        finally:
            self.lifecycle_events.append(
                (f"init/load_function_s{self.chunk_size}", time.perf_counter() - t0)
            )

    def _model_for_chunk_size(
        self,
        chunk_size: int,
        *,
        profiler: Optional[Any] = None,
        profile_prefix: str = "lm",
    ) -> ct.models.MLModel:
        model = self._models_by_chunk_size.get(chunk_size)
        if model is not None:
            if self.unload_inactive_functions:
                self._unload_inactive_function_handles(
                    active_chunk_size=chunk_size,
                    profile_prefix=profile_prefix,
                )
            return model

        name = self._function_names_by_chunk_size[chunk_size]
        event_name = f"{profile_prefix}/load_function_s{chunk_size}"
        t0 = time.perf_counter()
        with self._profile(profiler, event_name):
            model = load_coreml_model(
                self.model_path,
                compute_units=self.compute_units,
                function_name=name,
            )
        if profiler is None:
            self.lifecycle_events.append((event_name, time.perf_counter() - t0))
        if self.max_function_handles >= 1:
            if self.unload_inactive_functions:
                self._models_by_chunk_size[chunk_size] = model
                self._unload_inactive_function_handles(
                    active_chunk_size=chunk_size,
                    profile_prefix=profile_prefix,
                )
            else:
                loaded_extra = [
                    size
                    for size in self._models_by_chunk_size
                    if size != self.chunk_size
                ]
                while len(loaded_extra) >= self.max_function_handles:
                    evict = loaded_extra.pop(0)
                    self._unload_function(
                        evict,
                        event_name=f"{profile_prefix}/unload_function_s{evict}",
                    )
                self._models_by_chunk_size[chunk_size] = model
        return model

    def _unload_inactive_function_handles(
        self,
        *,
        active_chunk_size: int,
        profile_prefix: str,
    ) -> None:
        keep_sizes = {active_chunk_size}
        if self.keep_default_function_loaded:
            keep_sizes.add(self.chunk_size)
        for loaded_size in list(self._models_by_chunk_size):
            if loaded_size not in keep_sizes:
                self._unload_function(
                    loaded_size,
                    event_name=f"{profile_prefix}/unload_function_s{loaded_size}",
                )

    def _unload_function(
        self,
        chunk_size: int,
        *,
        event_name: Optional[str] = None,
    ) -> None:
        t0 = time.perf_counter()
        self._models_by_chunk_size.pop(chunk_size, None)
        self.lifecycle_events.append(
            (
                event_name or f"unload_function_s{chunk_size}",
                time.perf_counter() - t0,
            )
        )

    def reset_position(self) -> None:
        if self.is_chain:
            for submodel in self.submodels:
                submodel.reset_position()
            self.current_position = 0
            return
        self.current_position = 0

    def _ensure_persistent_state(
        self,
        model: Optional[ct.models.MLModel] = None,
    ) -> None:
        """Create the single request-reused MLState for this LM if needed."""
        if self.is_chain:
            for submodel in self.submodels:
                submodel._ensure_persistent_state()
            return
        if self.state is not None:
            return
        self.state = (model or self._active_or_default_model()).make_state()

    def drain_lifecycle_events(self) -> list[tuple[str, float]]:
        if self.is_chain:
            events: list[tuple[str, float]] = []
            for idx, submodel in enumerate(self.submodels):
                events.extend(
                    (f"part{idx}/{name}", duration)
                    for name, duration in submodel.drain_lifecycle_events()
                )
            return events
        events = self.lifecycle_events
        self.lifecycle_events = []
        return events

    @staticmethod
    @contextmanager
    def _profile(profiler: Optional[Any], name: str):
        if profiler is None:
            yield
            return
        if hasattr(profiler, "track"):
            with profiler.track(name):
                yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            profiler[name] = profiler.get(name, 0.0) + (time.perf_counter() - t0)

    def _position_id_value(self) -> np.ndarray:
        if self.position_id_shape == ():
            return np.array(self.current_position, dtype=np.int32)
        if self.position_id_shape == (1,):
            return np.array([self.current_position], dtype=np.int32)
        raise ValueError(f"unsupported position_id shape {self.position_id_shape}")

    @staticmethod
    def _description_for_function(spec: Any, function_name: Optional[str]) -> Any:
        functions = spec.description.functions
        if not functions:
            return spec.description
        target = function_name or spec.description.defaultFunctionName or "main"
        for function in functions:
            if function.name == target:
                return function
        raise ValueError(
            f"function {target!r} not found; available functions are "
            f"{[f.name for f in functions]}"
        )

    @staticmethod
    def _metadata_function_for_name(
        metadata: dict, function_name: Optional[str]
    ) -> dict:
        functions = metadata.get("functions", [])
        if not functions:
            return metadata
        target = function_name or "main"
        for function in functions:
            if function.get("name") == target:
                return function
        raise ValueError(
            f"function {target!r} not found; available functions are "
            f"{[f.get('name') for f in functions]}"
        )

    @staticmethod
    def _metadata_state_shapes(metadata: dict) -> list[tuple[str, tuple[int, ...]]]:
        for key in ("stateSchema", "states", "state"):
            schema = metadata.get(key)
            if isinstance(schema, list):
                shapes = []
                for item in schema:
                    if not isinstance(item, dict) or not item.get("shape"):
                        continue
                    shape = item["shape"]
                    if isinstance(shape, str):
                        shape = parse_shape_text(shape)
                    else:
                        shape = tuple(int(d) for d in shape)
                    shapes.append((item.get("name", ""), shape))
                return shapes
        return []

    @staticmethod
    def _spec_state_shapes(spec: Any) -> list[tuple[str, tuple[int, ...]]]:
        shapes = []
        for state in getattr(spec.description, "state", []):
            array_type = state.type.stateType.arrayType
            shapes.append((state.name, tuple(int(d) for d in array_type.shape)))
        return shapes

    @staticmethod
    def _cache_length_from_state_shapes(
        state_shapes: list[tuple[str, tuple[int, ...]]],
    ) -> int | None:
        lengths = [
            int(shape[2])
            for name, shape in state_shapes
            if any(cache_name in name for cache_name in ("key_cache", "value_cache"))
            and len(shape) >= 3
        ]
        return min(lengths) if lengths else None

    @staticmethod
    def _to_numpy(x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Return ``(float32_numpy, original_rank)``."""
        if not isinstance(x, np.ndarray):
            raise TypeError(f"inputs_embeds must be a numpy array, got {type(x)!r}")
        return x.astype(np.float32, copy=False), x.ndim

    @staticmethod
    def _to_nchw(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return np.ascontiguousarray(x[:, :, None, None], dtype=np.float32)
        if x.ndim == 3:
            return np.ascontiguousarray(
                np.transpose(x, (0, 2, 1))[:, :, None, :], dtype=np.float32
            )
        if x.ndim == 4:
            return np.ascontiguousarray(x, dtype=np.float32)
        raise ValueError(f"expected rank 2, 3, or 4 input, got shape {x.shape}")

    @staticmethod
    def _from_nchw(x: np.ndarray, original_rank: int) -> np.ndarray:
        if original_rank == 2:
            return np.ascontiguousarray(x[:, :, 0, 0])
        if original_rank == 3:
            return np.ascontiguousarray(np.transpose(x[:, :, 0, :], (0, 2, 1)))
        if original_rank == 4:
            return np.ascontiguousarray(x)
        raise ValueError(f"unsupported original rank {original_rank}")
