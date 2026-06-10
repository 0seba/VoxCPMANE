"""Pure numpy/CoreML VoxCPM2 generation path."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Callable, Generator, Sequence

import coremltools as ct
import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

from ._coreml_utils import (
    get_feature_info,
    load_coreml_model,
    resolve_model_path,
)
from .audio_vae_decoder import AudioVAEDecoder
from .audio_vae_encoder import AudioVAEEncoder
from .embeddings import load_embed_tokens
from .feat_encoder import FeatEncoder
from .lm import CoreMLMiniCPMLM
from .locdit import CoreMLUnifiedCFM


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
        feat_compute_units=ct.ComputeUnit.CPU_AND_NE,
        locdit_compute_units=ct.ComputeUnit.CPU_AND_NE,
        vae_encoder_compute_units=ct.ComputeUnit.CPU_ONLY,
        vae_decoder_compute_units=ct.ComputeUnit.CPU_ONLY,
        fsq_compute_units=ct.ComputeUnit.CPU_ONLY,
        proj_compute_units=ct.ComputeUnit.CPU_AND_NE,
        lm_prefill_chunk_size: int | None = None,
        lm_restrict_to_preload: bool = False,
        lm_startup_decode_warmup_repeats: int = 0,
        vae_early_decode_steps: int = 16,
        vae_batch_decode_steps: int = 4,
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
        self.embed_tokens = embed_tokens.astype(np.float16)
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

        def _model_path(
            override: Path | None, filename: str, *, use_compiled: bool = True
        ) -> Path:
            return resolve_model_path(
                override or mdir / filename,
                use_compiled=use_compiled,
                compiled_dir=compiled_dir,
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

        _compiled_sibling = lambda p: resolve_model_path(
            p, use_compiled=use_compiled_lm, compiled_dir=compiled_dir
        )

        lm_kwargs = {
            "compute_units": lm_compute_units,
            "unload_inactive_functions": lm_unload_inactive_functions,
            "idle_prefill_chunk_size": lm_idle_prefill_chunk_size,
            "preload_chunk_sizes": lm_preload_chunk_sizes,
            "keep_default_function_loaded": lm_keep_decode_function_loaded,
            "restrict_to_preload": lm_restrict_to_preload,
            "startup_decode_warmup_repeats": lm_startup_decode_warmup_repeats,
        }
        if base_lm_model_path is not None:
            self.base_lm = CoreMLMiniCPMLM(
                _compiled_sibling(base_lm_model_path), **lm_kwargs, label="base_lm"
            )
        elif base_lm_split_model_paths is not None:
            base_paths = [_compiled_sibling(p) for p in base_lm_split_model_paths]
            self.base_lm = CoreMLMiniCPMLM(
                base_paths,
                **lm_kwargs,
                label="base_lm",
            )
        elif base_lm_splits > 1:
            base_paths = [
                _compiled_sibling(
                    mdir
                    / f"base_lm_s{input_seq_length}_part{i}_of_{base_lm_splits}.mlpackage"
                )
                for i in range(base_lm_splits)
            ]
            self.base_lm = CoreMLMiniCPMLM(
                base_paths,
                **lm_kwargs,
                label="base_lm",
            )
        else:
            self.base_lm = CoreMLMiniCPMLM(
                _compiled_sibling(mdir / f"base_lm_s{input_seq_length}.mlpackage"),
                **lm_kwargs,
                label="base_lm",
            )

        self.residual_lm = CoreMLMiniCPMLM(
            _compiled_sibling(
                residual_lm_model_path
                or mdir / f"residual_lm_fused_s{input_seq_length}.mlpackage"
            ),
            **lm_kwargs,
            label="residual_lm",
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
        self.lm_cache_length = self._resolve_lm_cache_length()
        if hasattr(self.base_lm, "model_path"):
            base_lm_resolved = str(self.base_lm.model_path)
        elif hasattr(self.base_lm, "submodels"):
            base_lm_resolved = ", ".join(
                str(getattr(submodel, "model_path", "<unknown>"))
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

    def warmup_startup_models(self, repeats: int = 5) -> None:
        """Run synthetic predicts through each loaded CoreML model/function."""
        repeats = max(0, int(repeats))
        if repeats == 0:
            return

        t0 = time.perf_counter()
        print(
            f"🔥 CoreML startup warmup: {repeats} predict calls per loaded model/function...",
            flush=True,
        )

        def run(name: str, fn: Callable[[], None]) -> None:
            stage_t0 = time.perf_counter()
            fn()
            print(
                f"   ✓ {name}: {time.perf_counter() - stage_t0:.3f}s",
                flush=True,
            )

        def warm_vae_encoder() -> None:
            audio = np.zeros((self.vae_encoder.chunk_samples,), dtype=np.float32)
            self.vae_encoder.reset()
            for _ in range(repeats):
                self.vae_encoder.encode_chunk(audio)
            self.vae_encoder.reset()

        def warm_feat_encoder() -> None:
            patches = np.zeros(
                (1, 1, self.patch_size, self.latent_dim),
                dtype=np.float32,
            )
            for _ in range(repeats):
                self.feat_encoder.encode_patches(patches)

        def warm_base_lm() -> None:
            self.base_lm.warmup_loaded_functions(repeats=repeats)

        def warm_residual_lm() -> None:
            self.residual_lm.warmup_loaded_functions(repeats=repeats)

        def warm_locdit() -> None:
            mu = np.zeros((1, 2 * self.dit_hidden_dim), dtype=np.float32)
            cond = np.zeros(
                (1, self.latent_dim, self.patch_size),
                dtype=np.float32,
            )
            rng = np.random.default_rng(0)
            for _ in range(repeats):
                self.locdit.predict_numpy(
                    mu=mu,
                    n_timesteps=1,
                    patch_size=self.patch_size,
                    cond=cond,
                    cfg_value=1.0,
                    use_cfg_zero_star=False,
                    rng=rng,
                )

        def warm_vae_decoder() -> None:
            z = np.zeros(
                (1, self.latent_dim, self.patch_size),
                dtype=np.float32,
            )
            self.vae_decoder.reset()
            for _ in range(repeats):
                self.vae_decoder.decode_chunk(z)
            self.vae_decoder.reset()

        def warm_fsq() -> None:
            channels = getattr(self, "fsq_input_channels", self.hidden_size)
            x = np.zeros(
                (1, channels, 1, self._select_fsq_enumerated_chunk(1)),
                dtype=np.float32,
            )
            for _ in range(repeats):
                self._fsq(x, preferred_chunk_size=x.shape[-1])

        def warm_projections() -> None:
            lm_hidden = np.zeros((1, self.hidden_size, 1, 1), dtype=np.float32)
            residual_hidden = np.zeros((1, self.hidden_size, 1, 1), dtype=np.float32)
            for _ in range(repeats):
                self._projections(lm_hidden, residual_hidden)

        run("audio_vae_encoder", warm_vae_encoder)
        run("feat_encoder", warm_feat_encoder)
        run("base_lm loaded functions", warm_base_lm)
        run("residual_lm loaded functions", warm_residual_lm)
        run("locdit", warm_locdit)
        run("audio_vae_decoder", warm_vae_decoder)
        run("fsq", warm_fsq)
        run("projections", warm_projections)
        print(
            f"✅ CoreML startup warmup complete in {time.perf_counter() - t0:.3f}s.",
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

    def _resolve_lm_cache_length(self) -> int | None:
        lengths = [
            getattr(lm, "cache_length", None) for lm in (self.base_lm, self.residual_lm)
        ]
        lengths = [int(length) for length in lengths if length is not None]
        return min(lengths) if lengths else None

    def _limit_max_len_to_cache(self, max_len: int, prefill_tokens: int) -> int:
        max_len = int(max_len)
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        if self.lm_cache_length is None:
            return max_len

        available = int(self.lm_cache_length) - int(prefill_tokens)
        if available <= 0:
            raise ValueError(
                f"prompt uses {prefill_tokens} LM tokens, exceeding LM cache length "
                f"{self.lm_cache_length}"
            )
        return min(max_len, available)

    def preload_tokenizer(self, hf_model_id: str = "openbmb/VoxCPM2") -> None:
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = Tokenizer.from_pretrained(hf_model_id)
            self._unk_token_id = self._tokenizer.token_to_id("<unk>")
            self._multichar_chinese_tokens = {
                token
                for token in self._tokenizer.get_vocab().keys()
                if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
            }

    def _token_to_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            token_id = self._unk_token_id
        if token_id is None:
            raise KeyError(f"token {token!r} is not in the tokenizer vocabulary")
        return int(token_id)

    def _encode_text(self, text: str) -> np.ndarray:
        """Match upstream mask_multichar_chinese_tokens(tokenizer)(text)."""
        if not hasattr(self, "_tokenizer"):
            self.preload_tokenizer()
        processed_ids = []
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        for token, token_id in zip(encoding.tokens, encoding.ids):
            clean_token = token.replace("▁", "")
            if clean_token in self._multichar_chinese_tokens:
                processed_ids.extend(self._token_to_id(char) for char in clean_token)
            else:
                processed_ids.append(int(token_id))
        return np.array(processed_ids, dtype=np.int32)

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
        self.fsq_input_channels = self.hidden_size
        self.fsq_range_max_seq_length: int | None = None
        self.fsq_enumerated_seq_lengths: tuple[int, ...] = ()

        try:
            info = get_feature_info(self.fsq_model, model_path, "x")
        except KeyError:
            return

        self.fsq_input_dtype = info["dtype"]
        shape = info["shape"]
        if len(shape) == 4:
            self.fsq_input_channels = int(shape[1])
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
        self, preferred: int | None
    ) -> int:
        lengths = self.fsq_enumerated_seq_lengths
        if not lengths:
            return int(self.fsq_fixed_chunk_size)
        if preferred is not None and int(preferred) in lengths:
            return int(preferred)
        return int(lengths[0])

    def _fsq(
        self,
        x: np.ndarray,
        *,
        preferred_chunk_size: int | None = None,
    ) -> np.ndarray:
        """FSQ on NCHW tensor (1, C, 1, S) with chunked processing."""
        _, _, _, S = x.shape
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
            chunk = self._select_fsq_enumerated_chunk(preferred_chunk_size)
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

    def _audio_feature_from_source(
        self,
        *,
        feature: np.ndarray | None,
        wav_path: str,
        embed: np.ndarray | None,
        padding_mode: str,
    ) -> np.ndarray:
        if feature is not None:
            return np.asarray(feature, dtype=np.float32)
        if wav_path:
            return self.vae_encoder.encode_wav(wav_path, padding_mode=padding_mode)
        if embed is not None:
            return np.zeros(
                (int(np.asarray(embed).shape[0]), self.patch_size, self.latent_dim),
                dtype=np.float32,
            )
        return np.empty((0, self.patch_size, self.latent_dim), dtype=np.float32)

    def _embed_cache_or_none(
        self,
        name: str,
        embed: np.ndarray | None,
        feat: np.ndarray,
    ) -> np.ndarray | None:
        if embed is None:
            return None
        embed = np.asarray(embed, dtype=np.float32)
        expected = (feat.shape[0], self.hidden_size)
        if embed.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {embed.shape}")
        return embed

    def _coerce_prefix_feat_cond(
        self, value: np.ndarray | None
    ) -> np.ndarray | None:
        if value is None:
            return None
        cond = np.asarray(value, dtype=np.float32)
        if cond.shape == (self.patch_size, self.latent_dim):
            return cond.T.reshape(1, self.latent_dim, self.patch_size)
        if cond.shape == (self.latent_dim, self.patch_size):
            return cond.reshape(1, self.latent_dim, self.patch_size)
        if cond.shape == (1, self.latent_dim, self.patch_size):
            return cond
        raise ValueError(
            "prompt_prefix_feat_cond must have shape "
            f"({self.patch_size}, {self.latent_dim}), "
            f"({self.latent_dim}, {self.patch_size}), or "
            f"(1, {self.latent_dim}, {self.patch_size}); got {cond.shape}"
        )

    def _coerce_decode_context(self, value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        context = np.asarray(value, dtype=np.float32)
        if context.shape[1:] == (self.latent_dim, self.patch_size):
            return np.transpose(context, (0, 2, 1))
        if context.shape[1:] == (self.patch_size, self.latent_dim):
            return context
        raise ValueError(
            "prompt_decode_context must have shape "
            f"(N, {self.patch_size}, {self.latent_dim}) or "
            f"(N, {self.latent_dim}, {self.patch_size}); got {context.shape}"
        )

    def _prefill(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        reference_wav_path: str = "",
        prompt_audio_feat: np.ndarray | None = None,
        reference_audio_feat: np.ndarray | None = None,
        prompt_audio_embed: np.ndarray | None = None,
        reference_audio_embed: np.ndarray | None = None,
        prompt_prefix_feat_cond: np.ndarray | None = None,
        prompt_decode_context: np.ndarray | None = None,
        lm_prefix_cache_path: str | Path | None = None,
        lm_prefix_cache_read_paths: Sequence[str | Path] | None = None,
    ):
        """Build token/feature sequences and run LM prefill."""
        t_prefill = time.perf_counter()
        t_stage = t_prefill
        prefill_stage_seconds: dict[str, float] = {}

        def mark_prefill_stage(name: str) -> None:
            nonlocal t_stage
            now = time.perf_counter()
            prefill_stage_seconds[name] = now - t_stage
            t_stage = now

        prompt_text = prompt_text or ""
        prompt_wav_path = (prompt_wav_path or "").strip()
        reference_wav_path = (reference_wav_path or "").strip()

        has_prompt_audio = (
            prompt_audio_feat is not None
            or prompt_audio_embed is not None
            or bool(prompt_wav_path)
        )
        has_reference_audio = (
            reference_audio_feat is not None
            or reference_audio_embed is not None
            or bool(reference_wav_path)
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
        mark_prefill_stage("text_tokens")

        prompt_feat = self._audio_feature_from_source(
            feature=prompt_audio_feat,
            wav_path=prompt_wav_path,
            embed=prompt_audio_embed,
            padding_mode="left",
        )
        ref_feat = self._audio_feature_from_source(
            feature=reference_audio_feat,
            wav_path=reference_wav_path,
            embed=reference_audio_embed,
            padding_mode="right",
        )

        prompt_embed_cache = self._embed_cache_or_none(
            "prompt_audio_embed", prompt_audio_embed, prompt_feat
        )
        reference_embed_cache = self._embed_cache_or_none(
            "reference_audio_embed", reference_audio_embed, ref_feat
        )
        prefix_feat_cond_override = self._coerce_prefix_feat_cond(
            prompt_prefix_feat_cond
        )
        decode_context = self._coerce_decode_context(prompt_decode_context)
        mark_prefill_stage("audio_features")

        segments = []
        if has_reference_audio:
            ref_tokens, ref_feats, ref_text_mask, ref_feat_mask = self._make_ref_prefix(ref_feat)
            lm_prefix_length = len(ref_tokens)
            segments.append((ref_tokens, ref_feats, ref_text_mask, ref_feat_mask))
        else:
            lm_prefix_length = 0

        segments.append((
            text_token,
            np.zeros((text_length, self.patch_size, self.latent_dim), dtype=np.float32),
            np.ones(text_length, dtype=np.float32),
            np.zeros(text_length, dtype=np.float32),
        ))

        if has_prompt_audio:
            prompt_len = prompt_feat.shape[0]
            segments.append((
                np.zeros(prompt_len, dtype=np.int32),
                prompt_feat,
                np.zeros(prompt_len, dtype=np.float32),
                np.ones(prompt_len, dtype=np.float32),
            ))

        text_token = np.concatenate([s[0] for s in segments])
        audio_feat = np.concatenate([s[1] for s in segments], axis=0)
        text_mask_1d = np.concatenate([s[2] for s in segments])
        feat_mask_1d = np.concatenate([s[3] for s in segments])

        total_length = len(text_token)

        text_mask = text_mask_1d.reshape(1, 1, 1, total_length)
        feat_mask = feat_mask_1d.reshape(1, 1, 1, total_length)
        mark_prefill_stage("sequence_build")

        text_embed = self.embed_tokens[text_token] * self.scale_emb
        text_embed = text_embed.reshape(1, total_length, self.hidden_size).transpose(
            0, 2, 1
        )[:, :, None, :]
        mark_prefill_stage("text_embed")

        feat_indices = np.nonzero(feat_mask_1d > 0.0)[0]
        cached_audio_embeds = []
        if has_reference_audio and reference_embed_cache is not None:
            cached_audio_embeds.append(reference_embed_cache)
        if has_prompt_audio and prompt_embed_cache is not None:
            cached_audio_embeds.append(prompt_embed_cache)
        cached_audio_embed = (
            np.concatenate(cached_audio_embeds, axis=0) if cached_audio_embeds else None
        )
        if cached_audio_embed is not None and cached_audio_embed.shape == (
            len(feat_indices),
            self.hidden_size,
        ):
            feat_embed_tokens = np.zeros(
                (1, total_length, self.hidden_size), dtype=np.float32
            )
            feat_embed_tokens[:, feat_indices, :] = cached_audio_embed
            feat_embed = feat_embed_tokens.transpose(0, 2, 1)[:, :, None, :]
        elif len(feat_indices):
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
        mark_prefill_stage("feat_embed")

        combined = text_mask * text_embed + feat_mask * feat_embed

        if prefix_feat_cond_override is not None:
            prefix_feat_cond = prefix_feat_cond_override
        else:
            prefix_feat_cond = audio_feat[-1].T.reshape(
                1, self.latent_dim, self.patch_size
            )
        mark_prefill_stage("combine")

        prefix_cache_path = Path(lm_prefix_cache_path) if lm_prefix_cache_path else None
        prefix_cache_read_paths = [
            Path(path) for path in (lm_prefix_cache_read_paths or ())
        ]
        if prefix_cache_path is not None and prefix_cache_path not in prefix_cache_read_paths:
            prefix_cache_read_paths.append(prefix_cache_path)
        use_prefix_cache = (
            (prefix_cache_path is not None or bool(prefix_cache_read_paths))
            and lm_prefix_length > 0
            and lm_prefix_length < total_length
        )
        restored_prefix_cache = (
            self._restore_lm_prefix_cache_from_paths(
                prefix_cache_read_paths,
                lm_prefix_length,
            )
            if use_prefix_cache
            else False
        )
        mark_prefill_stage("lm_prefix_cache_restore")
        run_start = lm_prefix_length if restored_prefix_cache else 0
        combined_run = combined[..., run_start:]
        enc_outputs, _ = self.base_lm.forward(
            combined_run,
            is_causal=True,
            reset_state=not restored_prefix_cache,
            preferred_chunk_size=self._base_lm_prefill_chunk_size,
        )
        mark_prefill_stage("base_lm")

        run_feat_indices = feat_indices[feat_indices >= run_start] - run_start
        if len(run_feat_indices):
            fsq_out = self._fsq(
                enc_outputs[..., run_feat_indices],
                preferred_chunk_size=256,
            )
            enc_outputs = enc_outputs.copy()
            enc_outputs[..., run_feat_indices] = fsq_out
        mark_prefill_stage("fsq")
        lm_hidden = enc_outputs[..., -1:]

        feat_part = (feat_mask * feat_embed)[..., run_start:]
        residual_input = np.concatenate([enc_outputs, feat_part], axis=1)
        mark_prefill_stage("residual_prep")
        res_outputs, _ = self.residual_lm.forward(
            residual_input,
            is_causal=True,
            reset_state=not restored_prefix_cache,
            preferred_chunk_size=self._residual_lm_prefill_chunk_size,
        )
        residual_hidden = res_outputs[..., -1:]
        mark_prefill_stage("residual_lm")

        if use_prefix_cache and not restored_prefix_cache and prefix_cache_path is not None:
            self._save_lm_prefix_cache(prefix_cache_path, lm_prefix_length)
        mark_prefill_stage("lm_prefix_cache_save")

        dt = time.perf_counter() - t_prefill

        return {
            "lm_hidden": lm_hidden,
            "residual_hidden": residual_hidden,
            "prefix_feat_cond": prefix_feat_cond,
            "prefill_seconds": dt,
            "prefill_tokens": total_length,
            "prefill_stage_seconds": prefill_stage_seconds,
            "prefill_text_tokens": int(np.count_nonzero(text_mask_1d > 0.0)),
            "prefill_audio_tokens": int(len(feat_indices)),
            "prefill_reference_audio_tokens": int(ref_feat.shape[0]),
            "prefill_prompt_audio_tokens": int(prompt_feat.shape[0]),
            "prefill_lm_prefix_cache": bool(restored_prefix_cache),
            "prefill_lm_prefix_tokens": int(lm_prefix_length if use_prefix_cache else 0),
            "prompt_decode_context": decode_context,
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

    def _warm_vae_decoder(self, decode_context: np.ndarray | None) -> None:
        """Prime the streaming VAE decoder with prompt context and discard audio."""
        self.vae_decoder.reset()
        if decode_context is None:
            return
        context = np.asarray(decode_context, dtype=np.float32)
        if context.size == 0:
            return
        for patch in context:
            chunk = patch.T.reshape(1, self.latent_dim, self.patch_size)
            self.vae_decoder.decode_chunk(chunk)

    def _lm_snapshot_shapes(self, lm: object) -> dict[str, tuple[int, ...]]:
        if getattr(lm, "is_chain", False):
            merged: dict[str, tuple[int, ...]] = {}
            for submodel in getattr(lm, "submodels"):
                for name, shape in self._lm_snapshot_shapes(submodel).items():
                    if name not in merged:
                        merged[name] = shape
                        continue
                    previous = merged[name]
                    if previous[1:] != shape[1:]:
                        raise ValueError(
                            f"chain state {name!r} shapes cannot be merged: "
                            f"{previous} vs {shape}"
                        )
                    merged[name] = (previous[0] + shape[0], *previous[1:])
            return merged
        return {
            str(name): tuple(shape)
            for name, shape in getattr(lm, "state_shapes", [])
        }

    def _validate_lm_snapshot(
        self,
        lm: object,
        snapshot: dict[str, np.ndarray],
        *,
        label: str,
        prefix_length: int,
    ) -> None:
        expected = self._lm_snapshot_shapes(lm)
        if not expected:
            raise ValueError(f"{label} LM exposes no state shapes")
        missing = sorted(set(expected) - set(snapshot))
        extra = sorted(set(snapshot) - set(expected))
        if missing or extra:
            raise ValueError(
                f"{label} LM snapshot keys do not match current model; "
                f"missing={missing[:4]} extra={extra[:4]}"
            )
        for name, value in snapshot.items():
            shape = expected[name]
            arr = np.asarray(value)
            if len(shape) >= 3 and arr.ndim == len(shape) and shape[2] >= prefix_length:
                expected_shape = shape[:2] + (int(prefix_length),) + shape[3:]
            else:
                expected_shape = shape
            if tuple(arr.shape) != tuple(expected_shape):
                raise ValueError(
                    f"{label} LM state {name!r} has shape {arr.shape}; "
                    f"expected {expected_shape}"
                )

    def _save_lm_prefix_cache(self, cache_path: Path, prefix_length: int) -> None:
        """Save base/residual LM KV prefix slices as float16."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            arrays: dict[str, np.ndarray] = {}
            for k, v in self.base_lm.snapshot_state_prefix(prefix_length).items():
                arrays[f"base:{k}"] = np.asarray(v, dtype=np.float16)
            for k, v in self.residual_lm.snapshot_state_prefix(prefix_length).items():
                arrays[f"residual:{k}"] = np.asarray(v, dtype=np.float16)

            metadata = {
                "version": 2,
                "prefix_length": int(prefix_length),
            }
            tmp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
            np.savez_compressed(
                tmp_path,
                __metadata__=np.array(json.dumps(metadata)),
                **arrays,
            )
            tmp_path.replace(cache_path)
        except Exception as exc:
            print(f"⚠️  Failed to save LM prefix cache {cache_path}: {exc}", flush=True)

    def _restore_lm_prefix_cache(self, cache_path: Path, prefix_length: int) -> bool:
        """Restore base/residual LM KV prefix cache. Returns False on cache miss."""
        if not cache_path.exists():
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                metadata = json.loads(str(data["__metadata__"].item()))
                if int(metadata.get("prefix_length", -1)) != int(prefix_length):
                    return False
                if int(metadata.get("version", 1)) != 2:
                    return False
                base_snapshot = {}
                residual_snapshot = {}
                for key in data.files:
                    if key == "__metadata__":
                        continue
                    if key.startswith("base:"):
                        base_snapshot[key[5:]] = data[key]
                    elif key.startswith("residual:"):
                        residual_snapshot[key[9:]] = data[key]

                self._validate_lm_snapshot(
                    self.base_lm,
                    base_snapshot,
                    label="base",
                    prefix_length=prefix_length,
                )
                self._validate_lm_snapshot(
                    self.residual_lm,
                    residual_snapshot,
                    label="residual",
                    prefix_length=prefix_length,
                )
                self.base_lm.restore_state_prefix(base_snapshot, prefix_length)
                self.residual_lm.restore_state_prefix(residual_snapshot, prefix_length)
            return True
        except Exception as exc:
            print(f"⚠️  Ignoring invalid LM prefix cache {cache_path}: {exc}", flush=True)
            return False

    def _restore_lm_prefix_cache_from_paths(
        self,
        cache_paths: Sequence[Path],
        prefix_length: int,
    ) -> bool:
        for cache_path in cache_paths:
            if self._restore_lm_prefix_cache(cache_path, prefix_length):
                return True
        return False

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
            if hasattr(lm, "reset_position"):
                lm.reset_position()
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

    def _cleanup_prefill_functions(self) -> None:
        """Unload inactive LM function handles and run gc."""
        for name, lm in [("base_lm", self.base_lm), ("residual_lm", self.residual_lm)]:
            if not hasattr(lm, "chunk_size"):
                continue
            if not getattr(lm, "unload_inactive_functions", False):
                continue
            unloaded = False
            submodels = getattr(lm, "submodels", None)
            if submodels is not None:
                for submodel in submodels:
                    for loaded_size in list(
                        getattr(submodel, "_models_by_chunk_size", {})
                    ):
                        if loaded_size != submodel.chunk_size:
                            submodel._unload_function(
                                loaded_size,
                                event_name=f"decode/unload_function_s{loaded_size}",
                            )
                            unloaded = True
            else:
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
        prompt_audio_embed: np.ndarray | None = None,
        reference_audio_embed: np.ndarray | None = None,
        prompt_prefix_feat_cond: np.ndarray | None = None,
        prompt_decode_context: np.ndarray | None = None,
        lm_prefix_cache_path: str | Path | None = None,
        lm_prefix_cache_read_paths: Sequence[str | Path] | None = None,
        max_len: int = 256,
        min_len: int = 2,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate audio from text plus optional reference voice."""
        chunks = list(
            self.generate_streaming(
                target_text=target_text,
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
                reference_wav_path=reference_wav_path,
                prompt_audio_feat=prompt_audio_feat,
                reference_audio_feat=reference_audio_feat,
                prompt_audio_embed=prompt_audio_embed,
                reference_audio_embed=reference_audio_embed,
                prompt_prefix_feat_cond=prompt_prefix_feat_cond,
                prompt_decode_context=prompt_decode_context,
                lm_prefix_cache_path=lm_prefix_cache_path,
                lm_prefix_cache_read_paths=lm_prefix_cache_read_paths,
                max_len=max_len,
                min_len=min_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                seed=seed,
            )
        )
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def generate_streaming(
        self,
        target_text: str,
        prompt_text: str = "",
        reference_wav_path: str = "",
        prompt_wav_path: str = "",
        prompt_audio_feat: np.ndarray | None = None,
        reference_audio_feat: np.ndarray | None = None,
        prompt_audio_embed: np.ndarray | None = None,
        reference_audio_embed: np.ndarray | None = None,
        prompt_prefix_feat_cond: np.ndarray | None = None,
        prompt_decode_context: np.ndarray | None = None,
        lm_prefix_cache_path: str | Path | None = None,
        lm_prefix_cache_read_paths: Sequence[str | Path] | None = None,
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
            prompt_audio_embed=prompt_audio_embed,
            reference_audio_embed=reference_audio_embed,
            prompt_prefix_feat_cond=prompt_prefix_feat_cond,
            prompt_decode_context=prompt_decode_context,
            lm_prefix_cache_path=lm_prefix_cache_path,
            lm_prefix_cache_read_paths=lm_prefix_cache_read_paths,
        )
        max_len = self._limit_max_len_to_cache(max_len, state["prefill_tokens"])
        prefill_done_at = time.perf_counter()
        if metrics_callback is not None:
            metrics_callback(
                "prefill",
                {
                    "prefill_seconds": state["prefill_seconds"],
                    "prefill_tokens": state["prefill_tokens"],
                    "prefill_stage_seconds": state["prefill_stage_seconds"],
                    "prefill_text_tokens": state["prefill_text_tokens"],
                    "prefill_audio_tokens": state["prefill_audio_tokens"],
                    "prefill_reference_audio_tokens": state["prefill_reference_audio_tokens"],
                    "prefill_prompt_audio_tokens": state["prefill_prompt_audio_tokens"],
                    "prefill_lm_prefix_cache": state["prefill_lm_prefix_cache"],
                    "prefill_lm_prefix_tokens": state["prefill_lm_prefix_tokens"],
                    "at": prefill_done_at,
                },
            )
        self._swap_to_decode()
        swap_done_at = time.perf_counter()
        swap_to_decode_seconds = swap_done_at - prefill_done_at
        audio_samples = 0
        final_status = "completed"

        self._warm_vae_decoder(state.get("prompt_decode_context"))

        ar_step = 0
        pending_feats: list[np.ndarray] = []
        generation_start = time.perf_counter()
        if metrics_callback is not None:
            metrics_callback(
                "generation_start",
                {
                    "at": generation_start,
                    "swap_to_decode_seconds": swap_to_decode_seconds,
                },
            )
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
                if metrics_callback is not None:
                    metrics_callback(
                        "ar_loop",
                        {
                            "index": ar_step,
                            "at": time.perf_counter(),
                        },
                    )

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

            if pending_feats:
                batch = np.concatenate(pending_feats, axis=-1)
                audio_chunk = self.vae_decoder.decode_batch(batch).reshape(-1)
                audio_samples += int(audio_chunk.shape[0])
                yield audio_chunk
            if ar_step >= max_len:
                final_status = "max_len"
        except GeneratorExit:
            final_status = "stopped"
        except Exception:
            final_status = "failed"
            raise
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
