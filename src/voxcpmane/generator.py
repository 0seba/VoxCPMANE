"""Pure numpy/CoreML VoxCPM2 generation path."""

from __future__ import annotations

import gc
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Generator, Iterable

import coremltools as ct
import numpy as np
import soundfile as sf
from safetensors import safe_open

from ._coreml_utils import get_feature_info, is_compiled_model_path, load_coreml_model
from .audio_vae_decoder import AudioVAEDecoder
from .audio_vae_encoder import AudioVAEEncoder
from .feat_encoder import FeatEncoder
from .lm import CoreMLMiniCPMLM, CoreMLMiniCPMLMChain
from .locdit import CoreMLUnifiedCFM

DEFAULT_EMBEDDING_KEYS = (
    "tts_model.base_lm.embed_tokens.weight",
    "base_lm.embed_tokens.weight",
    "model.tts_model.base_lm.embed_tokens.weight",
    "lm.embed_tokens.weight",
    "embed_tokens.weight",
)


def _compiled_sibling_if_available(path: Path, *, use_compiled: bool = True) -> Path:
    compiled = path.with_suffix(".mlmodelc")
    if (
        use_compiled
        and compiled.exists()
        and (compiled / "metadata.json").exists()
        and (compiled / "model.mil").exists()
    ):
        return compiled
    return path


def _compiled_counterpart_in_dir(path: Path, compiled_dir: Path) -> Path:
    return compiled_dir / path.with_suffix(".mlmodelc").name


def _iter_safetensors_files(path: Path) -> Iterable[Path]:
    root = path
    if root.is_file():
        yield root
        return
    if not root.exists():
        raise FileNotFoundError(f"safetensors path does not exist: {root}")
    direct = sorted(root.glob("*.safetensors"))
    nested = (
        sorted((root / "snapshots").glob("*/*.safetensors"))
        if (root / "snapshots").exists()
        else []
    )
    for file in (*direct, *nested):
        yield file


def load_embed_tokens_from_safetensors(
    path: Path,
    *,
    key: str | None = None,
) -> np.ndarray:
    """Load the LM token embedding table from a safetensors file or directory."""

    keys = (key,) if key is not None else DEFAULT_EMBEDDING_KEYS
    suffixes = tuple(k for k in keys if k)
    candidates = list(_iter_safetensors_files(path))
    if not candidates:
        raise FileNotFoundError(f"no .safetensors files found under {path}")

    available: list[str] = []
    for file in candidates:
        with safe_open(str(file), framework="np") as tensors:
            file_keys = list(tensors.keys())
            available.extend(file_keys)
            selected = next((name for name in keys if name in file_keys), None)
            if selected is None and key is None:
                matches = [name for name in file_keys if name.endswith(suffixes)]
                if len(matches) == 1:
                    selected = matches[0]
            if selected is not None:
                try:
                    return tensors.get_tensor(selected).astype(np.float32)
                except TypeError as exc:
                    if "bfloat16" not in str(exc):
                        raise
                    try:
                        import ml_dtypes  # noqa: F401
                    except ImportError as import_exc:
                        raise TypeError(
                            f"embedding tensor {selected!r} in {file} uses "
                            "bfloat16; install ml-dtypes or run with "
                            "`uv run --extra conversion ...` after syncing."
                        ) from import_exc
                    return tensors.get_tensor(selected).astype(np.float32)

    preview = ", ".join(sorted(set(available))[:12])
    raise KeyError(
        f"could not find embedding tensor in safetensors files under {path}; "
        f"looked for {', '.join(keys)}. Available keys include: {preview}"
    )


def _resolve_hf_cache_dir(
    repo_id: str = "openbmb/VoxCPM2",
) -> Path | None:
    """Return the latest local HuggingFace snapshot for *repo_id*, if cached."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hf_cache / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(
        snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def load_embed_tokens(
    path: Path | None = None,
    *,
    key: str | None = None,
    hf_repo_id: str = "openbmb/VoxCPM2",
) -> np.ndarray:
    """Load the LM token embedding table from numpy, safetensors, or HF cache."""
    search_path = path

    if search_path is not None:
        if search_path.is_file() and search_path.suffix == ".npy":
            return np.load(str(search_path)).astype(np.float32)

        if search_path.is_dir():
            npy = search_path / "embed_tokens.npy"
            if npy.exists():
                return np.load(str(npy)).astype(np.float32)

    if search_path is not None:
        try:
            return load_embed_tokens_from_safetensors(search_path, key=key)
        except (FileNotFoundError, KeyError):
            pass  # fall through to HF cache

    hf_dir = _resolve_hf_cache_dir(hf_repo_id)
    if hf_dir is not None:
        try:
            return load_embed_tokens_from_safetensors(hf_dir, key=key)
        except (FileNotFoundError, KeyError):
            pass

    locations = [str(search_path)] if search_path else []
    if hf_dir:
        locations.append(f"HF cache ({hf_dir})")
    raise FileNotFoundError(
        "could not find embed_tokens — searched: "
        + ", ".join(locations or ["(no paths given)"])
    )


class VoxCPM2Generator:
    """Full VoxCPM2 TTS pipeline using only numpy + CoreML."""

    def __init__(
        self,
        model_dir: Path,
        *,
        embed_tokens: np.ndarray | None = None,
        embedding_safetensors_path: Path | None = None,
        embedding_key: str | None = None,
        scale_emb: float = 1.0,
        audio_start_token: int = 101,
        audio_end_token: int = 102,
        ref_audio_start_token: int = 103,
        ref_audio_end_token: int = 104,
        hidden_size: int = 2048,
        dit_hidden_dim: int = 1024,
        patch_size: int = 4,
        latent_dim: int = 64,
        vae_out_sample_rate: int = 48000,
        vae_decode_chunk_size: int = 1920,
        input_seq_length: int = 4,
        base_lm_splits: int = 2,
        base_lm_split_model_paths: list[Path] | None = None,
        base_lm_model_path: Path | None = None,
        residual_lm_model_path: Path | None = None,
        locdit_model_path: Path | None = None,
        vae_encoder_model_path: Path | None = None,
        feat_encoder_model_path: Path | None = None,
        vae_decoder_model_path: Path | None = None,
        fsq_model_path: Path | None = None,
        projections_model_path: Path | None = None,
        compiled_fallback_dir: Path | None = None,
        use_compiled_lm: bool = True,
        lm_unload_inactive_functions: bool = False,
        lm_idle_prefill_chunk_size: int | None = None,
        lm_preload_chunk_sizes: list[int] | None = None,
        lm_keep_decode_function_loaded: bool = False,
        # CoreML compute units
        lm_compute_units=ct.ComputeUnit.CPU_AND_NE,
        feat_compute_units=ct.ComputeUnit.ALL,
        locdit_compute_units=ct.ComputeUnit.CPU_AND_NE,
        vae_encoder_compute_units=ct.ComputeUnit.CPU_ONLY,
        vae_decoder_compute_units=ct.ComputeUnit.CPU_ONLY,
        fsq_compute_units=ct.ComputeUnit.CPU_ONLY,
        proj_compute_units=ct.ComputeUnit.CPU_AND_NE,
        lm_prefill_chunk_size: int | None = None,
        lm_async_decode_load: bool = False,
        lm_restrict_to_preload: bool = False,
        vae_early_decode_steps: int = 0,
        vae_batch_decode_steps: int = 1,
    ):
        mdir = model_dir
        compiled_dir = (
            compiled_fallback_dir.resolve()
            if compiled_fallback_dir is not None
            else None
        )
        if embed_tokens is None:
            embed_tokens = load_embed_tokens(
                embedding_safetensors_path or mdir,
                key=embedding_key,
            )
        self.embed_tokens = embed_tokens.astype(np.float32)
        self.scale_emb = scale_emb
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.ref_audio_start_token = ref_audio_start_token
        self.ref_audio_end_token = ref_audio_end_token
        self.hidden_size = hidden_size
        self.dit_hidden_dim = dit_hidden_dim
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.vae_out_sample_rate = vae_out_sample_rate
        self.vae_decode_chunk_size = vae_decode_chunk_size
        self.input_seq_length = input_seq_length
        self.lm_prefill_chunk_size = (
            int(lm_prefill_chunk_size) if lm_prefill_chunk_size is not None else None
        )
        self._base_lm_prefill_chunk_size = self.lm_prefill_chunk_size
        self._residual_lm_prefill_chunk_size = self.lm_prefill_chunk_size
        self.lm_async_decode_load = bool(lm_async_decode_load)
        self._decode_load_executor: ThreadPoolExecutor | None = None
        self._decode_load_future: Future | None = None

        def _resolve_model_path(path: Path, *, use_compiled: bool) -> Path:
            resolved = _compiled_sibling_if_available(path, use_compiled=use_compiled)
            if is_compiled_model_path(resolved):
                return resolved
            if use_compiled and compiled_dir is not None:
                fallback = _compiled_counterpart_in_dir(path, compiled_dir)
                if (
                    fallback.exists()
                    and (fallback / "metadata.json").exists()
                    and (fallback / "model.mil").exists()
                ):
                    return fallback
            return path

        def _model_path(
            override: Path | None, filename: str, *, use_compiled: bool = True
        ) -> Path:
            return _resolve_model_path(
                override or mdir / filename, use_compiled=use_compiled
            )

        # Load CoreML models
        vae_encoder_path = _model_path(
            vae_encoder_model_path, "audio_vae_encoder.mlpackage"
        )
        self.vae_encoder = AudioVAEEncoder(
            vae_encoder_path,
            chunk_samples=10240,
            compute_units=vae_encoder_compute_units,
        )
        feat_encoder_path = _model_path(
            feat_encoder_model_path, "feat_encoder.mlpackage"
        )
        self.feat_encoder = FeatEncoder(
            feat_encoder_path,
            compute_units=feat_compute_units,
        )

        def _compiled_sibling(path: Path) -> Path:
            return _resolve_model_path(path, use_compiled=use_compiled_lm)

        lm_kwargs = {
            "compute_units": lm_compute_units,
            "unload_inactive_functions": lm_unload_inactive_functions,
            "idle_prefill_chunk_size": lm_idle_prefill_chunk_size,
            "preload_chunk_sizes": lm_preload_chunk_sizes,
            "keep_default_function_loaded": lm_keep_decode_function_loaded,
            "restrict_to_preload": lm_restrict_to_preload,
        }
        if base_lm_model_path is not None:
            self.base_lm = CoreMLMiniCPMLM(
                _compiled_sibling(base_lm_model_path), **lm_kwargs
            )
        elif base_lm_split_model_paths is not None:
            base_paths = [_compiled_sibling(p) for p in base_lm_split_model_paths]
            self.base_lm = CoreMLMiniCPMLMChain(
                base_paths,
                compute_units=lm_compute_units,
            )
        elif base_lm_splits > 1:
            base_paths = [
                _compiled_sibling(
                    mdir
                    / f"base_lm_s{input_seq_length}_part{i}_of_{base_lm_splits}.mlpackage"
                )
                for i in range(base_lm_splits)
            ]
            self.base_lm = CoreMLMiniCPMLMChain(
                base_paths,
                compute_units=lm_compute_units,
            )
        else:
            self.base_lm = CoreMLMiniCPMLM(
                _compiled_sibling(mdir / f"base_lm_s{input_seq_length}.mlpackage"),
                **lm_kwargs,
            )

        self.residual_lm = CoreMLMiniCPMLM(
            _compiled_sibling(
                residual_lm_model_path
                or mdir / f"residual_lm_fused_s{input_seq_length}.mlpackage"
            ),
            **lm_kwargs,
        )
        self._base_lm_prefill_chunk_size = self._resolve_prefill_chunk_size(
            self.base_lm, self.lm_prefill_chunk_size
        )
        self._residual_lm_prefill_chunk_size = self._resolve_prefill_chunk_size(
            self.residual_lm, self.lm_prefill_chunk_size
        )

        locdit_resolved_path = _model_path(
            locdit_model_path, f"locdit_p{patch_size}_c{patch_size}.mlpackage"
        )
        self.locdit = CoreMLUnifiedCFM(
            locdit_resolved_path,
            in_channels=latent_dim,
            compute_units=locdit_compute_units,
        )

        vae_decoder_path = _model_path(
            vae_decoder_model_path, "audio_vae_decoder_lf4.mlpackage"
        )
        self.vae_decoder = AudioVAEDecoder(
            vae_decoder_path,
            latent_frames=patch_size,
            latent_dim=latent_dim,
            upsample_factor=vae_decode_chunk_size,
            out_sample_rate=vae_out_sample_rate,
            compute_units=vae_decoder_compute_units,
        )
        self.vae_early_decode_steps = int(vae_early_decode_steps)
        self.vae_batch_decode_steps = max(1, int(vae_batch_decode_steps))

        fsq_path = _model_path(fsq_model_path, "fsq_s4.mlpackage")
        projections_path = _model_path(projections_model_path, "projections.mlpackage")
        self.fsq_model = load_coreml_model(
            fsq_path,
            compute_units=fsq_compute_units,
        )
        self._configure_fsq_runtime(fsq_path)
        self.projections_model = load_coreml_model(
            projections_path,
            compute_units=proj_compute_units,
        )
        if hasattr(self.base_lm, "model_path"):
            base_lm_resolved = self.base_lm.model_path
        elif hasattr(self.base_lm, "submodels"):
            base_lm_resolved = ", ".join(
                getattr(submodel, "model_path", "<unknown>")
                for submodel in self.base_lm.submodels
            )
        else:
            base_lm_resolved = "<unknown>"
        print(
            "CoreML resolved model paths:\n"
            f"  vae_encoder: {vae_encoder_path}\n"
            f"  feat_encoder: {feat_encoder_path}\n"
            f"  base_lm: {base_lm_resolved}\n"
            f"  residual_lm: {self.residual_lm.model_path}\n"
            f"  locdit: {locdit_resolved_path}\n"
            f"  vae_decoder: {vae_decoder_path}\n"
            f"  fsq: {fsq_path}\n"
            f"  projections: {projections_path}",
            flush=True,
        )

    @staticmethod
    def _resolve_prefill_chunk_size(lm: object, preferred: int | None) -> int | None:
        if preferred is None:
            return None
        available = getattr(lm, "_function_names_by_chunk_size", None)
        if isinstance(available, dict) and preferred in available:
            return preferred
        return getattr(lm, "chunk_size", preferred)

    def preload_tokenizer(self, hf_model_id: str = "openbmb/VoxCPM2") -> None:
        if not hasattr(self, "_tokenizer"):
            from transformers import LlamaTokenizerFast

            try:
                self._tokenizer = LlamaTokenizerFast.from_pretrained(
                    hf_model_id,
                    local_files_only=True,
                )
            except Exception:
                self._tokenizer = LlamaTokenizerFast.from_pretrained(hf_model_id)
            self._multichar_chinese_tokens = {
                token
                for token in self._tokenizer.vocab.keys()
                if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
            }

    def _encode_text(self, text: str) -> np.ndarray:
        """Match upstream mask_multichar_chinese_tokens(tokenizer)(text)."""
        if not hasattr(self, "_tokenizer"):
            self.preload_tokenizer()
        processed = []
        for token in self._tokenizer.tokenize(text):
            clean_token = token.replace("▁", "")
            if clean_token in self._multichar_chinese_tokens:
                processed.extend(list(clean_token))
            else:
                processed.append(token)
        return np.array(
            self._tokenizer.convert_tokens_to_ids(processed), dtype=np.int32
        )

    @classmethod
    def from_voxcpm(cls, model, model_dir: Path, **kwargs) -> VoxCPM2Generator:
        """Create a generator from a loaded VoxCPM model."""
        tts = model.tts_model
        cfg = tts.config

        scale_emb = getattr(cfg.lm_config, "scale_emb", 1.0)
        use_mup = getattr(cfg.lm_config, "use_mup", False)
        if not use_mup:
            scale_emb = 1.0

        return cls(
            model_dir=model_dir,
            scale_emb=scale_emb,
            hidden_size=cfg.lm_config.hidden_size,
            dit_hidden_dim=cfg.dit_config.hidden_dim,
            patch_size=cfg.patch_size,
            latent_dim=tts.audio_vae.latent_dim,
            vae_out_sample_rate=int(tts.audio_vae.out_sample_rate),
            vae_decode_chunk_size=int(tts.audio_vae.decode_chunk_size),
            **kwargs,
        )

    def _configure_fsq_runtime(self, model_path: Path) -> None:
        self.fsq_input_dtype = np.float16
        self.fsq_fixed_chunk_size = self.input_seq_length
        self.fsq_range_max_seq_length: int | None = None
        self.fsq_enumerated_seq_lengths: tuple[int, ...] = ()

        try:
            info = get_feature_info(self.fsq_model, model_path, "x")
        except KeyError:
            return

        self.fsq_input_dtype = info["dtype"]
        shape = info["shape"]
        if len(shape) == 4:
            self.fsq_fixed_chunk_size = int(shape[-1])

        shape_range = info["shape_range"]
        if shape_range and len(shape_range) == 4:
            self.fsq_range_max_seq_length = int(shape_range[-1][1])
            return

        enum_shapes = info["enumerated_shapes"]
        lengths = sorted({int(s[-1]) for s in enum_shapes if len(s) == 4})
        if lengths:
            self.fsq_enumerated_seq_lengths = tuple(lengths)
            self.fsq_fixed_chunk_size = int(lengths[0])

    def _select_fsq_enumerated_chunk(
        self, preferred: int | None, remaining: int
    ) -> int:
        lengths = self.fsq_enumerated_seq_lengths
        if not lengths:
            return int(self.fsq_fixed_chunk_size)
        if preferred is not None and int(preferred) in lengths:
            return int(preferred)
        for length in sorted(lengths, reverse=True):
            if length <= remaining:
                return int(length)
        return int(lengths[0])

    def _fsq(
        self,
        x: np.ndarray,
        *,
        preferred_chunk_size: int | None = None,
    ) -> np.ndarray:
        """FSQ on NCHW tensor (1, C, 1, S) with chunked processing."""
        B, C, _, S = x.shape
        del B, C
        if self.fsq_range_max_seq_length is not None:
            max_chunk = int(self.fsq_range_max_seq_length)
            chunk = min(int(preferred_chunk_size or max_chunk), max_chunk)
            chunk = max(1, min(chunk, S))
            out_chunks = []
            for i in range(0, S, chunk):
                c = x[..., i : i + chunk].astype(self.fsq_input_dtype)
                out_chunks.append(self.fsq_model.predict({"x": c})["output"])
            return np.concatenate(out_chunks, axis=-1).astype(np.float32)

        if self.fsq_enumerated_seq_lengths:
            chunk = self._select_fsq_enumerated_chunk(preferred_chunk_size, S)
        else:
            chunk = int(self.fsq_fixed_chunk_size)

        out_chunks = []
        for i in range(0, S, chunk):
            c = x[..., i : i + chunk]
            pad = chunk - c.shape[-1]
            if pad:
                c = np.pad(c, ((0, 0), (0, 0), (0, 0), (0, pad)))
            c = c.astype(self.fsq_input_dtype)
            out_chunks.append(self.fsq_model.predict({"x": c})["output"])
        return np.concatenate(out_chunks, axis=-1)[..., :S].astype(np.float32)

    def _projections(self, lm_hidden: np.ndarray, residual_hidden: np.ndarray):
        """Returns (dit_hidden (1, 2*D, 1, 1), stop_flag (1, 2, 1, 1))."""
        out = self.projections_model.predict(
            {
                "lm_hidden": lm_hidden.astype(np.float16),
                "residual_hidden": residual_hidden.astype(np.float16),
            }
        )
        return out["dit_hidden"].astype(np.float32), out["stop_flag"].astype(np.float32)

    def _prefill(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        reference_wav_path: str = "",
        prompt_audio_feat: np.ndarray | None = None,
        reference_audio_feat: np.ndarray | None = None,
    ):
        """Build token/feature sequences and run LM prefill."""
        t_prefill = time.perf_counter()
        prompt_text = prompt_text or ""
        prompt_wav_path = (prompt_wav_path or "").strip()
        reference_wav_path = (reference_wav_path or "").strip()

        has_prompt_audio = prompt_audio_feat is not None or bool(prompt_wav_path)
        has_reference_audio = reference_audio_feat is not None or bool(
            reference_wav_path
        )
        has_prompt_text = bool(prompt_text)
        text = (
            prompt_text + target_text
            if has_prompt_audio and has_prompt_text
            else target_text
        )

        text_token = self._encode_text(text)
        text_token = np.concatenate([text_token, [self.audio_start_token]])
        text_length = len(text_token)

        if prompt_audio_feat is not None:
            prompt_feat = np.asarray(prompt_audio_feat, dtype=np.float32)
        elif prompt_wav_path:
            prompt_feat = self.vae_encoder.encode_wav(
                prompt_wav_path, padding_mode="left"
            )
        else:
            prompt_feat = np.empty(
                (0, self.patch_size, self.latent_dim), dtype=np.float32
            )

        if reference_audio_feat is not None:
            ref_feat = np.asarray(reference_audio_feat, dtype=np.float32)
        elif reference_wav_path:
            ref_feat = self.vae_encoder.encode_wav(
                reference_wav_path, padding_mode="right"
            )
        else:
            ref_feat = np.empty((0, self.patch_size, self.latent_dim), dtype=np.float32)

        for name, feat in (
            ("prompt_audio_feat", prompt_feat),
            ("reference_audio_feat", ref_feat),
        ):
            if feat.ndim != 3 or feat.shape[1:] != (self.patch_size, self.latent_dim):
                raise ValueError(
                    f"{name} must have shape (T, {self.patch_size}, {self.latent_dim}), "
                    f"got {feat.shape}"
                )

        text_pad_feat = np.zeros(
            (text_length, self.patch_size, self.latent_dim), dtype=np.float32
        )

        if has_reference_audio:
            ref_tokens, ref_feats, ref_text_mask, ref_feat_mask = self._make_ref_prefix(
                ref_feat
            )
            if has_prompt_audio:
                prompt_audio_length = prompt_feat.shape[0]
                prompt_pad_token = np.zeros(prompt_audio_length, dtype=np.int32)
                text_token = np.concatenate([ref_tokens, text_token, prompt_pad_token])
                audio_feat = np.concatenate(
                    [ref_feats, text_pad_feat, prompt_feat], axis=0
                )
                text_mask_1d = np.concatenate(
                    [
                        ref_text_mask,
                        np.ones(text_length, dtype=np.float32),
                        np.zeros(prompt_audio_length, dtype=np.float32),
                    ]
                )
                feat_mask_1d = np.concatenate(
                    [
                        ref_feat_mask,
                        np.zeros(text_length, dtype=np.float32),
                        np.ones(prompt_audio_length, dtype=np.float32),
                    ]
                )
            else:
                text_token = np.concatenate([ref_tokens, text_token])
                audio_feat = np.concatenate([ref_feats, text_pad_feat], axis=0)
                text_mask_1d = np.concatenate(
                    [
                        ref_text_mask,
                        np.ones(text_length, dtype=np.float32),
                    ]
                )
                feat_mask_1d = np.concatenate(
                    [
                        ref_feat_mask,
                        np.zeros(text_length, dtype=np.float32),
                    ]
                )
        elif has_prompt_audio:
            audio_length = prompt_feat.shape[0]
            prompt_pad_token = np.zeros(audio_length, dtype=np.int32)
            text_token = np.concatenate([text_token, prompt_pad_token])
            audio_feat = np.concatenate([text_pad_feat, prompt_feat], axis=0)
            text_mask_1d = np.concatenate(
                [
                    np.ones(text_length, dtype=np.float32),
                    np.zeros(audio_length, dtype=np.float32),
                ]
            )
            feat_mask_1d = np.concatenate(
                [
                    np.zeros(text_length, dtype=np.float32),
                    np.ones(audio_length, dtype=np.float32),
                ]
            )
        else:
            audio_feat = text_pad_feat
            text_mask_1d = np.ones(text_length, dtype=np.float32)
            feat_mask_1d = np.zeros(text_length, dtype=np.float32)

        total_length = len(text_token)

        text_mask = text_mask_1d.reshape(1, 1, 1, total_length)
        feat_mask = feat_mask_1d.reshape(1, 1, 1, total_length)

        text_embed = self.embed_tokens[text_token] * self.scale_emb
        text_embed = text_embed.reshape(1, total_length, self.hidden_size).transpose(
            0, 2, 1
        )[:, :, None, :]

        feat_indices = np.nonzero(feat_mask_1d > 0.0)[0]
        if len(feat_indices):
            feat_input = audio_feat[feat_indices].reshape(
                1, len(feat_indices), self.patch_size, self.latent_dim
            )
            feat_embed_tokens = np.zeros(
                (1, total_length, self.hidden_size), dtype=np.float32
            )
            feat_embed_tokens[:, feat_indices, :] = self.feat_encoder.encode_patches(
                feat_input,
                preferred_chunk_patches=getattr(
                    self.feat_encoder,
                    "prefill_chunk_patches",
                    None,
                ),
            )
            feat_embed = feat_embed_tokens.transpose(0, 2, 1)[:, :, None, :]
        else:
            feat_embed = np.zeros(
                (1, self.hidden_size, 1, total_length), dtype=np.float32
            )

        combined = text_mask * text_embed + feat_mask * feat_embed

        prefix_feat_cond = audio_feat[-1].T.reshape(1, self.latent_dim, self.patch_size)

        enc_outputs, _ = self.base_lm.forward(
            combined,
            is_causal=True,
            reset_state=True,
            preferred_chunk_size=self._base_lm_prefill_chunk_size,
        )

        if len(feat_indices):
            fsq_out = self._fsq(
                enc_outputs[..., feat_indices],
                preferred_chunk_size=256,
            )
            enc_outputs = enc_outputs.copy()
            enc_outputs[..., feat_indices] = fsq_out
        lm_hidden = enc_outputs[..., -1:]

        feat_part = feat_mask * feat_embed
        residual_input = np.concatenate([enc_outputs, feat_part], axis=1)
        res_outputs, _ = self.residual_lm.forward(
            residual_input,
            is_causal=True,
            reset_state=True,
            preferred_chunk_size=self._residual_lm_prefill_chunk_size,
        )
        residual_hidden = res_outputs[..., -1:]

        dt = time.perf_counter() - t_prefill

        return {
            "lm_hidden": lm_hidden,
            "residual_hidden": residual_hidden,
            "prefix_feat_cond": prefix_feat_cond,
            "prefill_seconds": dt,
            "prefill_tokens": total_length,
        }

    def _make_ref_prefix(self, ref_feat: np.ndarray):
        """Build upstream-compatible [ref_start ref_audio ref_end] prefix."""
        ref_len = ref_feat.shape[0]
        z1 = np.zeros((1, self.patch_size, self.latent_dim), dtype=np.float32)
        tokens = np.concatenate(
            [
                np.array([self.ref_audio_start_token], dtype=np.int32),
                np.zeros(ref_len, dtype=np.int32),
                np.array([self.ref_audio_end_token], dtype=np.int32),
            ]
        )
        feats = np.concatenate([z1, ref_feat, z1], axis=0)
        text_mask = np.concatenate(
            [
                np.ones(1, dtype=np.float32),
                np.zeros(ref_len, dtype=np.float32),
                np.ones(1, dtype=np.float32),
            ]
        )
        feat_mask = np.concatenate(
            [
                np.zeros(1, dtype=np.float32),
                np.ones(ref_len, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            ]
        )
        return tokens, feats, text_mask, feat_mask

    def _swap_to_idle(self) -> None:
        """Reload the prefill function and unload the decode function."""
        for name, lm in [("base_lm", self.base_lm), ("residual_lm", self.residual_lm)]:
            if not hasattr(lm, "idle_prefill_chunk_size"):
                continue
            if not getattr(lm, "unload_inactive_functions", False):
                continue
            idle = lm.idle_prefill_chunk_size
            if idle is None:
                continue
            unloaded = False
            if not getattr(lm, "keep_default_function_loaded", False):
                lm._unload_function(
                    lm.chunk_size,
                    event_name=f"idle/unload_function_s{lm.chunk_size}",
                )
                lm.model = None
                unloaded = True
            if unloaded:
                gc.collect()
            lm._model_for_chunk_size(idle, profile_prefix="idle")

    def _swap_to_decode(self) -> None:
        """Load the decode function and unload prefill."""
        self._ensure_decode_ready()
        self._cleanup_prefill_functions()

    def _ensure_decode_ready(self) -> None:
        """Ensure the decode function handle is loaded for both LMs."""
        for name, lm in [("base_lm", self.base_lm), ("residual_lm", self.residual_lm)]:
            if not hasattr(lm, "chunk_size"):
                continue
            cs = lm.chunk_size
            if cs not in lm._function_names_by_chunk_size:
                cs = min(lm._function_names_by_chunk_size)
            lm._model_for_chunk_size(cs, profile_prefix="decode")

    def _begin_decode_ready(self) -> Future | None:
        """Start decode-handle loading, optionally on a background thread."""
        if not self.lm_async_decode_load:
            self._ensure_decode_ready()
            return None
        if self._decode_load_executor is None:
            self._decode_load_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="lm-decode-load",
            )
        self._decode_load_future = self._decode_load_executor.submit(
            self._ensure_decode_ready
        )
        return self._decode_load_future

    def _finish_decode_ready(self) -> float:
        """Wait for a background decode load and return foreground wait time."""
        future = self._decode_load_future
        if future is None:
            return 0.0
        t0 = time.perf_counter()
        try:
            future.result()
        finally:
            self._decode_load_future = None
        return time.perf_counter() - t0

    def _cleanup_prefill_functions(self) -> None:
        """Unload inactive LM function handles and run gc."""
        for name, lm in [("base_lm", self.base_lm), ("residual_lm", self.residual_lm)]:
            if not hasattr(lm, "chunk_size"):
                continue
            if not getattr(lm, "unload_inactive_functions", False):
                continue
            unloaded = False
            for loaded_size in list(getattr(lm, "_models_by_chunk_size", {})):
                if loaded_size != lm.chunk_size:
                    lm._unload_function(
                        loaded_size,
                        event_name=f"decode/unload_function_s{loaded_size}",
                    )
                    unloaded = True
            if unloaded:
                gc.collect()

    def _ar_loop(
        self,
        state: dict,
        *,
        max_len: int,
        min_len: int,
        inference_timesteps: int,
        cfg_value: float,
        rng: np.random.Generator,
    ) -> Generator[np.ndarray, None, None]:
        """Core autoregressive loop yielding predicted features."""
        lm_hidden = state["lm_hidden"]
        residual_hidden = state["residual_hidden"]
        prefix_feat_cond = state["prefix_feat_cond"]

        for i in range(max_len):
            dit_hidden, stop_flag = self._projections(lm_hidden, residual_hidden)
            pred_feat = self.locdit.predict_numpy(
                mu=dit_hidden.reshape(1, -1),
                n_timesteps=inference_timesteps,
                patch_size=self.patch_size,
                cond=prefix_feat_cond,
                cfg_value=cfg_value,
                rng=rng,
            )

            yield pred_feat

            stop = int(np.argmax(stop_flag.reshape(-1)))
            if i > min_len and stop == 1:
                break

            pf = pred_feat.transpose(0, 2, 1)
            pf = pf.reshape(1, 1, self.patch_size, self.latent_dim)
            curr_embed = self.feat_encoder.encode_patches(pf)
            curr_embed_nchw = curr_embed.transpose(0, 2, 1)[:, :, None, :]
            prefix_feat_cond = pred_feat

            lm_hidden = self.base_lm.forward_step(curr_embed_nchw, None)
            lm_hidden = self._fsq(lm_hidden)

            residual_input = np.concatenate([lm_hidden, curr_embed_nchw], axis=1)
            residual_hidden = self.residual_lm.forward_step(residual_input, None)

    def generate(
        self,
        target_text: str,
        prompt_text: str = "",
        reference_wav_path: str = "",
        prompt_wav_path: str = "",
        prompt_audio_feat: np.ndarray | None = None,
        reference_audio_feat: np.ndarray | None = None,
        max_len: int = 256,
        min_len: int = 2,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate audio from text plus optional reference voice."""
        rng = np.random.default_rng(seed)

        state = self._prefill(
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            reference_wav_path=reference_wav_path,
            prompt_audio_feat=prompt_audio_feat,
            reference_audio_feat=reference_audio_feat,
        )
        self._swap_to_decode()

        pred_feat_seq = []
        try:
            for pred_feat in self._ar_loop(
                state,
                max_len=max_len,
                min_len=min_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                rng=rng,
            ):
                pred_feat_seq.append(pred_feat)
        finally:
            self._swap_to_idle()

        all_feats = np.concatenate(pred_feat_seq, axis=-1)
        latent = all_feats[0]

        self.vae_decoder.reset()
        upsample = self.vae_decode_chunk_size
        total_P = latent.shape[-1]
        audio_parts = []

        for start in range(0, total_P, self.patch_size):
            chunk = latent[:, start : start + self.patch_size].reshape(
                1, self.latent_dim, self.patch_size
            )
            audio_parts.append(self.vae_decoder.decode_chunk(chunk).reshape(-1))

        return np.concatenate(audio_parts)[: total_P * upsample]

    def generate_streaming(
        self,
        target_text: str,
        prompt_text: str = "",
        reference_wav_path: str = "",
        prompt_wav_path: str = "",
        prompt_audio_feat: np.ndarray | None = None,
        reference_audio_feat: np.ndarray | None = None,
        max_len: int = 256,
        min_len: int = 2,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        seed: int | None = None,
        metrics_callback: Callable[[str, dict], None] | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """Like :meth:`generate`, but yields audio chunks during generation."""
        rng = np.random.default_rng(seed)

        state = self._prefill(
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            reference_wav_path=reference_wav_path,
            prompt_audio_feat=prompt_audio_feat,
            reference_audio_feat=reference_audio_feat,
        )
        prefill_done_at = time.perf_counter()
        if metrics_callback is not None:
            metrics_callback(
                "prefill",
                {
                    "prefill_seconds": float(state.get("prefill_seconds", 0.0)),
                    "prefill_tokens": int(state.get("prefill_tokens", 0)),
                    "at": prefill_done_at,
                },
            )
        self._begin_decode_ready()
        swap_done_at = time.perf_counter()
        swap_to_decode_seconds = swap_done_at - prefill_done_at
        generation_start = swap_done_at
        audio_samples = 0
        final_status = "completed"
        prefill_cleanup_done = False
        if metrics_callback is not None:
            metrics_callback(
                "generation_start",
                {
                    "at": generation_start,
                    "swap_to_decode_seconds": swap_to_decode_seconds,
                },
            )

        self.vae_decoder.reset()

        ar_step = 0
        pending_feats: list[np.ndarray] = []
        try:
            for pred_feat in self._ar_loop(
                state,
                max_len=max_len,
                min_len=min_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                rng=rng,
            ):
                ar_step += 1

                if (
                    ar_step <= self.vae_early_decode_steps
                    or self.vae_batch_decode_steps <= 1
                ):
                    audio_chunk = self.vae_decoder.decode_chunk(pred_feat).reshape(-1)
                    audio_samples += int(audio_chunk.shape[0])
                    yield audio_chunk
                else:
                    pending_feats.append(pred_feat)
                    if len(pending_feats) >= self.vae_batch_decode_steps:
                        batch = np.concatenate(pending_feats, axis=-1)
                        audio_chunk = self.vae_decoder.decode_batch(batch).reshape(-1)
                        audio_samples += int(audio_chunk.shape[0])
                        pending_feats = []
                        yield audio_chunk

                if not prefill_cleanup_done:
                    prefill_cleanup_done = True
                    decode_wait_seconds = self._finish_decode_ready()
                    if metrics_callback is not None:
                        metrics_callback(
                            "decode_ready",
                            {
                                "at": time.perf_counter(),
                                "decode_load_wait_seconds": decode_wait_seconds,
                            },
                        )
                    self._cleanup_prefill_functions()

            if pending_feats:
                batch = np.concatenate(pending_feats, axis=-1)
                audio_chunk = self.vae_decoder.decode_batch(batch).reshape(-1)
                audio_samples += int(audio_chunk.shape[0])
                yield audio_chunk
        except GeneratorExit:
            final_status = "stopped"
        finally:
            if metrics_callback is not None:
                metrics_callback(
                    "final",
                    {
                        "status": final_status,
                        "generation_seconds": time.perf_counter() - generation_start,
                        "audio_seconds": audio_samples
                        / float(self.vae_out_sample_rate),
                    },
                )
            self._swap_to_idle()

    def generate_to_file(
        self,
        output_path: str,
        **kwargs,
    ) -> tuple[np.ndarray, float]:
        """Generate and save to WAV. Returns (audio, elapsed_seconds)."""
        t0 = time.perf_counter()
        audio = self.generate(**kwargs)
        dt = time.perf_counter() - t0
        sf.write(output_path, audio, self.vae_out_sample_rate)
        return audio, dt
