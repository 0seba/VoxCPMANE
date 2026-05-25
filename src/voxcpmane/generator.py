"""Pure numpy/CoreML VoxCPM2 generation path."""

from __future__ import annotations

import gc
import json
import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Generator, Sequence

import coremltools as ct
import numpy as np
import soundfile as sf

from ._coreml_utils import (
    get_feature_info,
    load_coreml_model,
    resolve_model_path,
)
from .audio_vae_decoder import AudioVAEDecoder
from .audio_vae_encoder import AudioVAEEncoder
from .embeddings import load_embed_tokens, load_embed_tokens_from_safetensors
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
        lm_async_decode_load: bool = False,
        lm_async_prefill_unload: bool = False,
        lm_restrict_to_preload: bool = False,
        prefill_audio_async: bool = False,
        prefill_audio_queue_size: int = 2,
        vae_early_decode_steps: int = 16,
        vae_batch_decode_steps: int = 4,
        vae_async_decode: bool = False,
        vae_decode_max_pending: int = 2,
    ):
        mdir = model_dir
        self.model_dir = mdir
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
        self.lm_async_decode_load = bool(lm_async_decode_load)
        self.lm_async_prefill_unload = bool(lm_async_prefill_unload)
        self._decode_load_executor: ThreadPoolExecutor | None = None
        self._decode_load_futures: dict[str, Future] = {}
        self._prefill_cleanup_executor: ThreadPoolExecutor | None = None
        self._prefill_cleanup_future: Future | None = None
        self.prefill_audio_async = bool(prefill_audio_async)
        self.prefill_audio_queue_size = max(1, int(prefill_audio_queue_size))

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
        }
        if base_lm_model_path is not None:
            self.base_lm = CoreMLMiniCPMLM(
                _compiled_sibling(base_lm_model_path), **lm_kwargs
            )
        elif base_lm_split_model_paths is not None:
            base_paths = [_compiled_sibling(p) for p in base_lm_split_model_paths]
            self.base_lm = CoreMLMiniCPMLM(
                base_paths,
                **lm_kwargs,
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
        self.vae_async_decode = bool(vae_async_decode)
        self.vae_decode_max_pending = max(1, int(vae_decode_max_pending))

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
            try:
                from tokenizers import Tokenizer
            except ModuleNotFoundError as exc:
                if exc.name != "tokenizers":
                    raise
                raise RuntimeError(
                    "Text tokenization requires the optional 'tokenizer' extra: "
                    "install voxcpmane2[tokenizer], or pass token IDs directly "
                    "when using the benchmark path."
                ) from exc

            tokenizer_file = self.model_dir / "tokenizer.json"
            if tokenizer_file.exists():
                self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
            else:
                self._tokenizer = Tokenizer.from_pretrained(hf_model_id)

            self._multichar_chinese_tokens = {
                token
                for token in self._tokenizer.get_vocab().keys()
                if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
            }

    def _encode_text(self, text: str) -> np.ndarray:
        """Match upstream mask_multichar_chinese_tokens(tokenizer)(text)."""
        if not hasattr(self, "_tokenizer"):
            self.preload_tokenizer()
        processed = []
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        for token in encoding.tokens:
            clean_token = token.replace("▁", "")
            if clean_token in self._multichar_chinese_tokens:
                processed.extend(list(clean_token))
            else:
                processed.append(token)
        token_ids = []
        for token in processed:
            token_id = self._tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"tokenizer has no ID for token {token!r}")
            token_ids.append(token_id)
        return np.array(token_ids, dtype=np.int32)

    @staticmethod
    def _coerce_token_ids(token_ids: Sequence[int] | np.ndarray | None) -> np.ndarray | None:
        if token_ids is None:
            return None
        ids = np.asarray(token_ids, dtype=np.int32).reshape(-1)
        if ids.size == 0:
            raise ValueError("token ID sequence must not be empty")
        return ids

    def _build_text_token_ids(
        self,
        *,
        target_text: str,
        prompt_text: str,
        has_prompt_audio: bool,
        has_prompt_text: bool,
        target_token_ids: Sequence[int] | np.ndarray | None,
        prompt_token_ids: Sequence[int] | np.ndarray | None,
    ) -> np.ndarray:
        target_ids = self._coerce_token_ids(target_token_ids)
        prompt_ids = self._coerce_token_ids(prompt_token_ids)
        if target_ids is not None:
            if has_prompt_audio and has_prompt_text:
                if prompt_ids is None:
                    prompt_ids = self._encode_text(prompt_text)
                text_token = np.concatenate([prompt_ids, target_ids])
            else:
                text_token = target_ids
        else:
            text = (
                prompt_text + target_text
                if has_prompt_audio and has_prompt_text
                else target_text
            )
            text_token = self._encode_text(text)
        return np.concatenate([text_token, [self.audio_start_token]])

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

    def _make_audio_source(
        self,
        *,
        name: str,
        feature: np.ndarray | None,
        wav_path: str,
        embed: np.ndarray | None,
        padding_mode: str,
    ) -> dict:
        if feature is not None:
            feat = np.asarray(feature, dtype=np.float32)
            return {
                "name": name,
                "kind": "feature",
                "feature": feat,
                "embed": self._embed_cache_or_none(name, embed, feat),
                "count": int(feat.shape[0]),
            }
        if wav_path:
            audio, target_patches = self.vae_encoder.prepare_wav(
                Path(wav_path),
                padding_mode=padding_mode,
            )
            return {
                "name": name,
                "kind": "wav",
                "audio": audio,
                "target_patches": int(target_patches),
                "embed": None,
                "count": int(target_patches),
            }
        if embed is not None:
            embed_arr = np.asarray(embed, dtype=np.float32)
            feat = np.zeros(
                (int(embed_arr.shape[0]), self.patch_size, self.latent_dim),
                dtype=np.float32,
            )
            return {
                "name": name,
                "kind": "feature",
                "feature": feat,
                "embed": self._embed_cache_or_none(name, embed_arr, feat),
                "count": int(feat.shape[0]),
            }
        return {
            "name": name,
            "kind": "empty",
            "feature": np.empty((0, self.patch_size, self.latent_dim), dtype=np.float32),
            "embed": None,
            "count": 0,
        }

    def _iter_wav_source_chunks(self, source: dict):
        audio = np.asarray(source["audio"], dtype=np.float32)
        target_patches = int(source["target_patches"])
        if not self.prefill_audio_async:
            yield from self.vae_encoder.iter_padded_audio_patch_chunks(
                audio,
                target_patches,
            )
            return

        out_queue: queue.Queue = queue.Queue(maxsize=self.prefill_audio_queue_size)

        def produce() -> None:
            try:
                for chunk in self.vae_encoder.iter_padded_audio_patch_chunks(
                    audio,
                    target_patches,
                ):
                    out_queue.put(chunk)
            except BaseException as exc:
                out_queue.put(exc)
            finally:
                out_queue.put(None)

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"vae-encode-{source['name']}",
        ) as executor:
            future = executor.submit(produce)
            try:
                while True:
                    item = out_queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
                future.result()
            finally:
                if not future.done():
                    future.cancel()

    def _iter_audio_feature_chunks(self, source: dict):
        count = int(source["count"])
        if count <= 0:
            return
        embed = source.get("embed")
        if embed is not None:
            embed_arr = np.asarray(embed, dtype=np.float32)
            chunk_size = self._base_lm_prefill_chunk_size or self.input_seq_length
            chunk_size = max(1, int(chunk_size))
            for start in range(0, count, chunk_size):
                end = min(start + chunk_size, count)
                feat = np.zeros(
                    (end - start, self.patch_size, self.latent_dim),
                    dtype=np.float32,
                )
                yield feat, embed_arr[start:end]
            return
        if source["kind"] == "wav":
            for feat in self._iter_wav_source_chunks(source):
                yield np.asarray(feat, dtype=np.float32), None
            return
        feat = np.asarray(source["feature"], dtype=np.float32)
        chunk_size = self._base_lm_prefill_chunk_size or feat.shape[0] or 1
        chunk_size = max(1, int(chunk_size))
        for start in range(0, feat.shape[0], chunk_size):
            yield feat[start : start + chunk_size], None

    def _text_prefill_block(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        tokens = np.asarray(tokens, dtype=np.int32)
        length = int(tokens.shape[0])
        text_embed = self.embed_tokens[tokens] * self.scale_emb
        combined = text_embed.reshape(1, length, self.hidden_size).transpose(0, 2, 1)
        combined = combined[:, :, None, :].astype(np.float32, copy=False)
        feat_part = np.zeros_like(combined, dtype=np.float32)
        return combined, feat_part, False

    def _audio_prefill_block(
        self,
        feat: np.ndarray,
        embed: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        feat = np.asarray(feat, dtype=np.float32)
        if feat.shape[0] == 0:
            empty = np.zeros((1, self.hidden_size, 1, 0), dtype=np.float32)
            return empty, empty, True
        if embed is not None:
            feat_embed_tokens = np.asarray(embed, dtype=np.float32).reshape(
                1,
                feat.shape[0],
                self.hidden_size,
            )
        else:
            feat_embed_tokens = self.feat_encoder.encode_patches(
                feat.reshape(1, feat.shape[0], self.patch_size, self.latent_dim),
                preferred_chunk_patches=getattr(
                    self.feat_encoder,
                    "prefill_chunk_patches",
                    None,
                ),
            )
        combined = feat_embed_tokens.transpose(0, 2, 1)[:, :, None, :]
        combined = combined.astype(np.float32, copy=False)
        return combined, combined, True

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

    def _prefill_streaming_audio(
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
        target_token_ids: Sequence[int] | np.ndarray | None = None,
        prompt_token_ids: Sequence[int] | np.ndarray | None = None,
    ):
        """Prefill path that can overlap CPU VAE encoding with NE prefill work."""
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
        text_token = self._build_text_token_ids(
            target_text=target_text,
            prompt_text=prompt_text,
            has_prompt_audio=has_prompt_audio,
            has_prompt_text=has_prompt_text,
            target_token_ids=target_token_ids,
            prompt_token_ids=prompt_token_ids,
        )
        text_length = len(text_token)
        mark_prefill_stage("text_tokens")

        prompt_source = self._make_audio_source(
            name="prompt_audio_embed",
            feature=prompt_audio_feat,
            wav_path=prompt_wav_path,
            embed=prompt_audio_embed,
            padding_mode="left",
        )
        ref_source = self._make_audio_source(
            name="reference_audio_embed",
            feature=reference_audio_feat,
            wav_path=reference_wav_path,
            embed=reference_audio_embed,
            padding_mode="right",
        )
        prefix_feat_cond_override = self._coerce_prefix_feat_cond(
            prompt_prefix_feat_cond
        )
        decode_context = self._coerce_decode_context(prompt_decode_context)
        mark_prefill_stage("audio_features")

        total_length = text_length
        lm_prefix_length = 0
        if has_reference_audio:
            lm_prefix_length = int(ref_source["count"]) + 2
            total_length += lm_prefix_length
        if has_prompt_audio:
            total_length += int(prompt_source["count"])
        mark_prefill_stage("sequence_build")

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
        sequence_offset = 0
        base_blocks: list[tuple[np.ndarray, np.ndarray]] = []
        last_prompt_patch: np.ndarray | None = None
        lm_hidden: np.ndarray | None = None
        base_has_run = False
        base_decode_load_started = False
        prefill_chunk_size = int(
            self._base_lm_prefill_chunk_size or self.input_seq_length
        )
        pending_combined: list[np.ndarray] = []
        pending_feat_part: list[np.ndarray] = []
        pending_audio_mask: list[np.ndarray] = []

        def pending_length() -> int:
            return sum(int(block.shape[-1]) for block in pending_combined)

        def flush_pending(final: bool = False) -> None:
            nonlocal base_decode_load_started, base_has_run, lm_hidden
            total = pending_length()
            if total <= 0:
                return
            flush_len = total if final else (total // prefill_chunk_size) * prefill_chunk_size
            if flush_len <= 0:
                return

            combined_all = np.concatenate(pending_combined, axis=-1)
            feat_part_all = np.concatenate(pending_feat_part, axis=-1)
            audio_mask_all = np.concatenate(pending_audio_mask, axis=0)

            combined = combined_all[..., :flush_len]
            feat_part = feat_part_all[..., :flush_len]
            audio_mask = audio_mask_all[:flush_len]
            remainder = combined_all.shape[-1] - flush_len
            pending_combined.clear()
            pending_feat_part.clear()
            pending_audio_mask.clear()
            if remainder:
                pending_combined.append(combined_all[..., flush_len:])
                pending_feat_part.append(feat_part_all[..., flush_len:])
                pending_audio_mask.append(audio_mask_all[flush_len:])

            enc_outputs, _ = self.base_lm.forward(
                combined,
                is_causal=True,
                reset_state=(not restored_prefix_cache and not base_has_run),
                preferred_chunk_size=self._base_lm_prefill_chunk_size,
            )
            base_has_run = True
            if final and not base_decode_load_started:
                self._begin_lm_decode_ready("base_lm", self.base_lm)
                base_decode_load_started = True
            audio_indices = np.nonzero(audio_mask > 0.0)[0]
            if len(audio_indices):
                enc_for_residual = enc_outputs.copy()
                fsq_out = self._fsq(
                    enc_outputs[..., audio_indices],
                    preferred_chunk_size=256,
                )
                enc_for_residual[..., audio_indices] = fsq_out
            else:
                enc_for_residual = enc_outputs
            lm_hidden = enc_for_residual[..., -1:]
            base_blocks.append((enc_for_residual, feat_part))

        def consume_block(
            combined: np.ndarray,
            feat_part: np.ndarray,
            is_audio: bool,
        ) -> None:
            nonlocal sequence_offset
            block_len = int(combined.shape[-1])
            if block_len == 0:
                return
            block_start = sequence_offset
            block_end = block_start + block_len
            sequence_offset = block_end
            if block_end <= run_start:
                return
            if block_start < run_start:
                keep_from = run_start - block_start
                combined = combined[..., keep_from:]
                feat_part = feat_part[..., keep_from:]
                block_len = int(combined.shape[-1])
            pending_combined.append(combined)
            pending_feat_part.append(feat_part)
            pending_audio_mask.append(
                np.full(block_len, 1.0 if is_audio else 0.0, dtype=np.float32)
            )
            flush_pending(final=False)

        if has_reference_audio:
            start_token = np.array([self.ref_audio_start_token], dtype=np.int32)
            consume_block(*self._text_prefill_block(start_token))
            for feat_chunk, embed_chunk in self._iter_audio_feature_chunks(ref_source):
                consume_block(*self._audio_prefill_block(feat_chunk, embed_chunk))
            end_token = np.array([self.ref_audio_end_token], dtype=np.int32)
            consume_block(*self._text_prefill_block(end_token))

        consume_block(*self._text_prefill_block(text_token))

        if has_prompt_audio:
            for feat_chunk, embed_chunk in self._iter_audio_feature_chunks(prompt_source):
                if feat_chunk.shape[0]:
                    last_prompt_patch = feat_chunk[-1]
                consume_block(*self._audio_prefill_block(feat_chunk, embed_chunk))

        flush_pending(final=True)
        mark_prefill_stage("stream_base_lm")
        if not base_decode_load_started:
            self._begin_lm_decode_ready("base_lm", self.base_lm)

        residual_hidden: np.ndarray | None = None
        first_residual_block = True
        for enc_for_residual, feat_part in base_blocks:
            residual_input = np.concatenate([enc_for_residual, feat_part], axis=1)
            res_outputs, _ = self.residual_lm.forward(
                residual_input,
                is_causal=True,
                reset_state=first_residual_block and not restored_prefix_cache,
                preferred_chunk_size=self._residual_lm_prefill_chunk_size,
            )
            first_residual_block = False
            residual_hidden = res_outputs[..., -1:]
        mark_prefill_stage("residual_lm")
        self._begin_lm_decode_ready("residual_lm", self.residual_lm)

        if lm_hidden is None or residual_hidden is None:
            raise RuntimeError("prefill produced no LM outputs")

        if prefix_feat_cond_override is not None:
            prefix_feat_cond = prefix_feat_cond_override
        elif last_prompt_patch is not None:
            prefix_feat_cond = last_prompt_patch.T.reshape(
                1, self.latent_dim, self.patch_size
            )
        else:
            prefix_feat_cond = np.zeros(
                (1, self.latent_dim, self.patch_size),
                dtype=np.float32,
            )

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
            "prefill_text_tokens": int(text_length + (2 if has_reference_audio else 0)),
            "prefill_audio_tokens": int(
                (ref_source["count"] if has_reference_audio else 0)
                + (prompt_source["count"] if has_prompt_audio else 0)
            ),
            "prefill_reference_audio_tokens": int(
                ref_source["count"] if has_reference_audio else 0
            ),
            "prefill_prompt_audio_tokens": int(
                prompt_source["count"] if has_prompt_audio else 0
            ),
            "prefill_lm_prefix_cache": bool(restored_prefix_cache),
            "prefill_lm_prefix_tokens": int(lm_prefix_length if use_prefix_cache else 0),
            "prompt_decode_context": decode_context,
        }

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
        target_token_ids: Sequence[int] | np.ndarray | None = None,
        prompt_token_ids: Sequence[int] | np.ndarray | None = None,
    ):
        """Build token/feature sequences and run LM prefill."""
        if self.prefill_audio_async and (prompt_wav_path or reference_wav_path):
            return self._prefill_streaming_audio(
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
                target_token_ids=target_token_ids,
                prompt_token_ids=prompt_token_ids,
            )
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
        text_token = self._build_text_token_ids(
            target_text=target_text,
            prompt_text=prompt_text,
            has_prompt_audio=has_prompt_audio,
            has_prompt_text=has_prompt_text,
            target_token_ids=target_token_ids,
            prompt_token_ids=prompt_token_ids,
        )
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
        self._begin_lm_decode_ready("base_lm", self.base_lm)

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
        self._begin_lm_decode_ready("residual_lm", self.residual_lm)

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
            shapes = {}
            for idx, submodel in enumerate(getattr(lm, "submodels")):
                for name, shape in self._lm_snapshot_shapes(submodel).items():
                    shapes[f"part_{idx}:{name}"] = shape
            return shapes
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
        def swap_lm_to_idle(name: str, lm: object) -> None:
            if getattr(lm, "is_chain", False):
                for idx, submodel in enumerate(getattr(lm, "submodels")):
                    swap_lm_to_idle(f"{name}/part{idx}", submodel)
                return
            if not hasattr(lm, "idle_prefill_chunk_size"):
                return
            if not getattr(lm, "unload_inactive_functions", False):
                return
            idle = lm.idle_prefill_chunk_size
            if idle is None:
                return
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

        for name, lm in self._iter_lms():
            swap_lm_to_idle(name, lm)

    def _swap_to_decode(self) -> None:
        """Load the decode function and unload prefill."""
        self._ensure_decode_ready()
        self._cleanup_prefill_functions()

    def _iter_lms(self):
        return (("base_lm", self.base_lm), ("residual_lm", self.residual_lm))

    @staticmethod
    def _decode_chunk_size_for_lm(lm: object) -> int | None:
        if not hasattr(lm, "chunk_size"):
            return None
        cs = int(lm.chunk_size)
        function_names = getattr(lm, "_function_names_by_chunk_size", {})
        if cs not in function_names and function_names:
            cs = min(function_names)
        return cs

    def _ensure_lm_decode_ready(self, label: str, lm: object) -> None:
        """Ensure one LM has its decode handle loaded."""
        if getattr(lm, "is_chain", False):
            for idx, submodel in enumerate(getattr(lm, "submodels")):
                self._ensure_lm_decode_ready(f"{label}/part{idx}", submodel)
            return
        cs = self._decode_chunk_size_for_lm(lm)
        if cs is None:
            return
        lm._model_for_chunk_size(cs, profile_prefix=f"decode/{label}")

    def _begin_lm_decode_ready(self, label: str, lm: object) -> Future | None:
        """Start one LM decode-handle load, optionally in the background."""
        if not self.lm_async_decode_load:
            return None
        future = self._decode_load_futures.get(label)
        if future is not None and not future.done():
            return future
        if future is not None and future.done():
            return future
        if self._decode_load_executor is None:
            self._decode_load_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="lm-decode-load",
            )
        future = self._decode_load_executor.submit(
            self._ensure_lm_decode_ready,
            label,
            lm,
        )
        self._decode_load_futures[label] = future
        return future

    def _ensure_decode_ready(self) -> None:
        """Ensure the decode function handle is loaded for both LMs."""
        for name, lm in self._iter_lms():
            self._ensure_lm_decode_ready(name, lm)

    def _begin_decode_ready(self) -> Future | None:
        """Start decode-handle loading, optionally on a background thread."""
        if not self.lm_async_decode_load:
            self._ensure_decode_ready()
            return None
        futures = [
            self._begin_lm_decode_ready(name, lm)
            for name, lm in self._iter_lms()
        ]
        return next((future for future in futures if future is not None), None)

    def _finish_decode_ready(self) -> dict[str, float]:
        """Wait for background decode loads and return foreground wait time by LM."""
        waits: dict[str, float] = {}
        for label in list(self._decode_load_futures):
            waits[label] = self._finish_lm_decode_ready(label)
        return waits

    def _finish_lm_decode_ready(self, label: str) -> float:
        """Wait for one background decode load and return foreground wait time."""
        future = self._decode_load_futures.get(label)
        if future is None:
            return 0.0
        t0 = time.perf_counter()
        try:
            future.result()
        finally:
            self._decode_load_futures.pop(label, None)
        return time.perf_counter() - t0

    def _begin_prefill_cleanup(self) -> Future | None:
        """Unload inactive prefill handles, optionally in the background."""
        future = self._prefill_cleanup_future
        if future is not None and not future.done():
            return future
        if self.lm_async_prefill_unload:
            if self._prefill_cleanup_executor is None:
                self._prefill_cleanup_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="lm-prefill-unload",
                )
            self._prefill_cleanup_future = self._prefill_cleanup_executor.submit(
                self._cleanup_prefill_functions
            )
            return self._prefill_cleanup_future
        self._cleanup_prefill_functions()
        return None

    def _finish_prefill_cleanup(self) -> float:
        """Wait for a background prefill unload and return foreground wait time."""
        future = self._prefill_cleanup_future
        if future is None:
            return 0.0
        t0 = time.perf_counter()
        try:
            future.result()
        finally:
            self._prefill_cleanup_future = None
        return time.perf_counter() - t0

    def _cleanup_prefill_functions(self) -> None:
        """Unload inactive LM function handles and run gc."""
        def cleanup_lm(name: str, lm: object) -> None:
            if getattr(lm, "is_chain", False):
                for idx, submodel in enumerate(getattr(lm, "submodels")):
                    cleanup_lm(f"{name}/part{idx}", submodel)
                return
            if not hasattr(lm, "chunk_size"):
                return
            if not getattr(lm, "unload_inactive_functions", False):
                return
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

        for name, lm in self._iter_lms():
            cleanup_lm(name, lm)

    def _ar_loop(
        self,
        state: dict,
        *,
        max_len: int,
        min_len: int,
        inference_timesteps: int,
        cfg_value: float,
        rng: np.random.Generator,
        decode_ready_callback: Callable[[str, float], None] | None = None,
        iteration_callback: Callable[[str, dict], None] | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """Core autoregressive loop yielding predicted features."""
        lm_hidden = state["lm_hidden"]
        residual_hidden = state["residual_hidden"]
        prefix_feat_cond = state["prefix_feat_cond"]
        base_decode_ready = False
        residual_decode_ready = False

        def finish_decode(label: str) -> None:
            wait_seconds = self._finish_lm_decode_ready(label)
            if decode_ready_callback is not None:
                decode_ready_callback(label, wait_seconds)

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

            if not base_decode_ready:
                finish_decode("base_lm")
                base_decode_ready = True
            lm_hidden = self.base_lm.forward_step(curr_embed_nchw, None)
            lm_hidden = self._fsq(lm_hidden)

            residual_input = np.concatenate([lm_hidden, curr_embed_nchw], axis=1)
            if not residual_decode_ready:
                finish_decode("residual_lm")
                residual_decode_ready = True
            residual_hidden = self.residual_lm.forward_step(residual_input, None)
            if i == 0 and iteration_callback is not None:
                iteration_callback("first_ar_iteration", {"at": time.perf_counter()})

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
        target_token_ids: Sequence[int] | np.ndarray | None = None,
        prompt_token_ids: Sequence[int] | np.ndarray | None = None,
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
                target_token_ids=target_token_ids,
                prompt_token_ids=prompt_token_ids,
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
        target_token_ids: Sequence[int] | np.ndarray | None = None,
        prompt_token_ids: Sequence[int] | np.ndarray | None = None,
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
            target_token_ids=target_token_ids,
            prompt_token_ids=prompt_token_ids,
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
        self._begin_decode_ready()
        swap_done_at = time.perf_counter()
        swap_to_decode_seconds = swap_done_at - prefill_done_at
        generation_start = swap_done_at
        audio_samples = 0
        final_status = "completed"
        prefill_cleanup_started = False
        if self.lm_async_prefill_unload and not self._decode_load_futures:
            self._begin_prefill_cleanup()
            prefill_cleanup_started = True

        decode_ready_waits: dict[str, float] = {}
        first_ar_iteration_reported = False

        def mark_decode_ready(label: str, wait_seconds: float) -> None:
            nonlocal prefill_cleanup_started
            if label in decode_ready_waits:
                return
            decode_ready_waits[label] = float(wait_seconds)
            if metrics_callback is not None:
                metrics_callback(
                    "decode_ready",
                    {
                        "at": time.perf_counter(),
                        "decode_load_wait_seconds": float(wait_seconds),
                        "decode_load_wait_by_lm": {label: float(wait_seconds)},
                    },
                )
            if (
                label == "residual_lm"
                and not prefill_cleanup_started
                and not self._decode_load_futures
            ):
                self._begin_prefill_cleanup()
                prefill_cleanup_started = True

        def mark_iteration(kind: str, values: dict) -> None:
            nonlocal first_ar_iteration_reported
            if kind != "first_ar_iteration" or first_ar_iteration_reported:
                return
            first_ar_iteration_reported = True
            at = float(values.get("at", time.perf_counter()))
            if metrics_callback is not None:
                metrics_callback(
                    "first_ar_iteration",
                    {
                        "at": at,
                        "first_ar_iteration_seconds": at - generation_start,
                    },
                )

        if metrics_callback is not None:
            metrics_callback(
                "generation_start",
                {
                    "at": generation_start,
                    "swap_to_decode_seconds": swap_to_decode_seconds,
                },
            )

        self._warm_vae_decoder(state.get("prompt_decode_context"))

        ar_step = 0
        pending_feats: list[np.ndarray] = []
        pending_vae_decodes: list[Future] = []
        vae_decode_executor: ThreadPoolExecutor | None = None
        first_audio_released = False

        def prepare_audio_chunk_for_release(audio_chunk: np.ndarray) -> np.ndarray:
            nonlocal audio_samples, first_audio_released
            if not first_audio_released:
                wait_seconds = self._finish_lm_decode_ready("base_lm")
                mark_decode_ready("base_lm", wait_seconds)
                first_audio_released = True
            audio_samples += int(audio_chunk.shape[0])
            return audio_chunk

        try:
            for pred_feat in self._ar_loop(
                state,
                max_len=max_len,
                min_len=min_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                rng=rng,
                decode_ready_callback=mark_decode_ready,
                iteration_callback=mark_iteration,
            ):
                ar_step += 1

                if self.vae_async_decode and ar_step > self.vae_early_decode_steps:
                    if vae_decode_executor is None:
                        vae_decode_executor = ThreadPoolExecutor(
                            max_workers=1,
                            thread_name_prefix="vae-decode",
                        )
                    pending_vae_decodes.append(
                        vae_decode_executor.submit(
                            self.vae_decoder.decode_chunk,
                            pred_feat.copy(),
                        )
                    )
                    if len(pending_vae_decodes) >= self.vae_decode_max_pending:
                        audio_chunk = pending_vae_decodes.pop(0).result().reshape(-1)
                        yield prepare_audio_chunk_for_release(audio_chunk)
                else:
                    if (
                        ar_step <= self.vae_early_decode_steps
                        or self.vae_batch_decode_steps <= 1
                    ):
                        audio_chunk = self.vae_decoder.decode_chunk(pred_feat).reshape(-1)
                        yield prepare_audio_chunk_for_release(audio_chunk)
                    else:
                        pending_feats.append(pred_feat)
                        if len(pending_feats) >= self.vae_batch_decode_steps:
                            batch = np.concatenate(pending_feats, axis=-1)
                            audio_chunk = self.vae_decoder.decode_batch(batch).reshape(-1)
                            pending_feats = []
                            yield prepare_audio_chunk_for_release(audio_chunk)

            while pending_vae_decodes:
                audio_chunk = pending_vae_decodes.pop(0).result().reshape(-1)
                yield prepare_audio_chunk_for_release(audio_chunk)
            if pending_feats:
                batch = np.concatenate(pending_feats, axis=-1)
                audio_chunk = self.vae_decoder.decode_batch(batch).reshape(-1)
                yield prepare_audio_chunk_for_release(audio_chunk)
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
            if self._decode_load_futures:
                for label, wait_seconds in self._finish_decode_ready().items():
                    mark_decode_ready(label, wait_seconds)
            cleanup_wait_seconds = self._finish_prefill_cleanup()
            if metrics_callback is not None and cleanup_wait_seconds > 0.0:
                metrics_callback(
                    "prefill_cleanup",
                    {
                        "at": time.perf_counter(),
                        "prefill_cleanup_wait_seconds": cleanup_wait_seconds,
                    },
                )
            if vae_decode_executor is not None:
                vae_decode_executor.shutdown(wait=True, cancel_futures=True)
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
