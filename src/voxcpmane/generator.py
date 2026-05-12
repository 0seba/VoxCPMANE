"""Pure numpy/CoreML VoxCPM2 generation path — no PyTorch at runtime.

Uses pre-converted CoreML packages for all neural network inference.
Only numpy and the tokenizer library are needed for orchestration.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Generator, Iterable

import coremltools as ct
import numpy as np
import soundfile as sf
from safetensors import safe_open

log = logging.getLogger("voxcpmane.generator")

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


def _load_coreml_model(path: Path, *, compute_units: ct.ComputeUnit):
    if path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(path), compute_units=compute_units)
    return ct.models.MLModel(str(path), compute_units=compute_units)


def _iter_safetensors_files(path: str | os.PathLike[str]) -> Iterable[Path]:
    root = Path(path)
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
    path: str | os.PathLike[str],
    *,
    key: str | None = None,
) -> np.ndarray:
    """Load the LM token embedding table from a safetensors file or directory."""

    keys = (key,) if key is not None else DEFAULT_EMBEDDING_KEYS
    suffixes = tuple(k for k in keys if k)
    candidates = list(_iter_safetensors_files(path))
    if not candidates:
        raise FileNotFoundError(f"no .safetensors files found under {Path(path)}")

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
        f"could not find embedding tensor in safetensors files under {Path(path)}; "
        f"looked for {', '.join(keys)}. Available keys include: {preview}"
    )


def _resolve_hf_cache_dir(
    repo_id: str = "openbmb/VoxCPM2",
) -> Path | None:
    """Find a local HuggingFace Hub cache directory for *repo_id*.

    Returns the latest snapshot directory, or ``None`` if not cached.
    """
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hf_cache / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    # Pick the most-recently-modified snapshot
    candidates = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_embed_tokens(
    path: str | os.PathLike[str] | None = None,
    *,
    key: str | None = None,
    hf_repo_id: str = "openbmb/VoxCPM2",
) -> np.ndarray:
    """Load the LM token embedding table from numpy or safetensors.

    Resolution order:

    1. If *path* points to a ``.npy`` file, load it directly.
    2. If *path* is a directory, look for ``embed_tokens.npy`` inside it.
    3. If *path* contains ``.safetensors`` files, search them for the
       embedding key (see :func:`load_embed_tokens_from_safetensors`).
    4. If none of the above worked and *path* was None, try the local
       HuggingFace cache for *hf_repo_id*.
    """
    search_path = Path(path) if path is not None else None

    # --- .npy path ---
    if search_path is not None:
        if search_path.is_file() and search_path.suffix == ".npy":
            return np.load(str(search_path)).astype(np.float32)

        if search_path.is_dir():
            npy = search_path / "embed_tokens.npy"
            if npy.exists():
                return np.load(str(npy)).astype(np.float32)

    # --- safetensors path ---
    if search_path is not None:
        try:
            return load_embed_tokens_from_safetensors(search_path, key=key)
        except (FileNotFoundError, KeyError):
            pass  # fall through to HF cache

    # --- HuggingFace cache fallback ---
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
        model_dir: str,
        *,
        embed_tokens: np.ndarray | None = None,
        embedding_safetensors_path: str | os.PathLike[str] | None = None,
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
        base_lm_split_model_paths: list[str] | None = None,
        base_lm_model_path: str | os.PathLike[str] | None = None,
        residual_lm_model_path: str | os.PathLike[str] | None = None,
        locdit_model_path: str | os.PathLike[str] | None = None,
        vae_encoder_model_path: str | os.PathLike[str] | None = None,
        feat_encoder_model_path: str | os.PathLike[str] | None = None,
        vae_decoder_model_path: str | os.PathLike[str] | None = None,
        fsq_model_path: str | os.PathLike[str] | None = None,
        projections_model_path: str | os.PathLike[str] | None = None,
        compiled_fallback_dir: str | os.PathLike[str] | None = None,
        use_compiled_lm: bool = True,
        lm_unload_inactive_functions: bool = False,
        lm_idle_prefill_chunk_size: int | None = None,
        lm_preload_chunk_sizes: list[int] | None = None,
        # CoreML compute units
        lm_compute_units=ct.ComputeUnit.CPU_AND_NE,
        feat_compute_units=ct.ComputeUnit.ALL,
        locdit_compute_units=ct.ComputeUnit.CPU_AND_NE,
        vae_encoder_compute_units=ct.ComputeUnit.CPU_ONLY,
        vae_decoder_compute_units=ct.ComputeUnit.CPU_ONLY,
        fsq_compute_units=ct.ComputeUnit.CPU_ONLY,
        proj_compute_units=ct.ComputeUnit.CPU_AND_NE,
        lm_prefill_chunk_size: int | None = None,
        debug: bool = False,
    ):
        self.debug = debug
        if self.debug:
            logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")
            log.setLevel(logging.DEBUG)
            log.debug("VoxCPM2Generator.__init__ starting")
        mdir = Path(model_dir)
        compiled_dir = (
            Path(compiled_fallback_dir).resolve()
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
        self.hidden_size = hidden_size
        self.dit_hidden_dim = dit_hidden_dim
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.vae_out_sample_rate = vae_out_sample_rate
        self.vae_decode_chunk_size = vae_decode_chunk_size
        self.input_seq_length = input_seq_length
        self.lm_prefill_chunk_size = (
            int(lm_prefill_chunk_size)
            if lm_prefill_chunk_size is not None
            else None
        )
        self._base_lm_prefill_chunk_size = self.lm_prefill_chunk_size
        self._residual_lm_prefill_chunk_size = self.lm_prefill_chunk_size

        def _resolve_model_path(path: Path, *, use_compiled: bool) -> Path:
            resolved = _compiled_sibling_if_available(path, use_compiled=use_compiled)
            if resolved.suffix == ".mlmodelc":
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

        # Load CoreML models
        vae_encoder_src = (
            Path(vae_encoder_model_path)
            if vae_encoder_model_path is not None
            else mdir / "audio_vae_encoder.mlpackage"
        )
        vae_encoder_path = _resolve_model_path(vae_encoder_src, use_compiled=True)
        self.vae_encoder = AudioVAEEncoder(
            str(vae_encoder_path),
            chunk_samples=10240,
            compute_units=vae_encoder_compute_units,
        )
        feat_encoder_src = (
            Path(feat_encoder_model_path)
            if feat_encoder_model_path is not None
            else mdir / "feat_encoder.mlpackage"
        )
        feat_encoder_path = _resolve_model_path(feat_encoder_src, use_compiled=True)
        self.feat_encoder = FeatEncoder(
            str(feat_encoder_path),
            compute_units=feat_compute_units,
        )

        def _compiled_sibling(path: Path) -> Path:
            return _resolve_model_path(path, use_compiled=use_compiled_lm)

        if base_lm_model_path is not None:
            self.base_lm = CoreMLMiniCPMLM(
                str(_compiled_sibling(Path(base_lm_model_path))),
                compute_units=lm_compute_units,
                unload_inactive_functions=lm_unload_inactive_functions,
                idle_prefill_chunk_size=lm_idle_prefill_chunk_size,
                preload_chunk_sizes=lm_preload_chunk_sizes,
                debug=self.debug,
            )
        elif base_lm_split_model_paths is not None:
            base_paths = [str(_compiled_sibling(Path(p))) for p in base_lm_split_model_paths]
            self.base_lm = CoreMLMiniCPMLMChain(
                base_paths, compute_units=lm_compute_units,
            )
        elif base_lm_splits > 1:
            base_paths = [
                str(_compiled_sibling(mdir / f"base_lm_s{input_seq_length}_part{i}_of_{base_lm_splits}.mlpackage"))
                for i in range(base_lm_splits)
            ]
            self.base_lm = CoreMLMiniCPMLMChain(
                base_paths, compute_units=lm_compute_units,
            )
        else:
            self.base_lm = CoreMLMiniCPMLM(
                str(_compiled_sibling(mdir / f"base_lm_s{input_seq_length}.mlpackage")),
                compute_units=lm_compute_units,
                unload_inactive_functions=lm_unload_inactive_functions,
                idle_prefill_chunk_size=lm_idle_prefill_chunk_size,
                preload_chunk_sizes=lm_preload_chunk_sizes,
                debug=self.debug,
            )

        residual_path = (
            Path(residual_lm_model_path)
            if residual_lm_model_path is not None
            else mdir / f"residual_lm_fused_s{input_seq_length}.mlpackage"
        )
        self.residual_lm = CoreMLMiniCPMLM(
            str(_compiled_sibling(residual_path)),
            compute_units=lm_compute_units,
            unload_inactive_functions=lm_unload_inactive_functions,
            idle_prefill_chunk_size=lm_idle_prefill_chunk_size,
            preload_chunk_sizes=lm_preload_chunk_sizes,
            debug=self.debug,
        )
        self._base_lm_prefill_chunk_size = self._resolve_prefill_chunk_size(
            self.base_lm, self.lm_prefill_chunk_size
        )
        self._residual_lm_prefill_chunk_size = self._resolve_prefill_chunk_size(
            self.residual_lm, self.lm_prefill_chunk_size
        )
        if self.debug and (
            self._base_lm_prefill_chunk_size != self.lm_prefill_chunk_size
            or self._residual_lm_prefill_chunk_size != self.lm_prefill_chunk_size
        ):
            log.debug(
                "prefill chunk override: requested=%s base_lm=%s residual_lm=%s",
                self.lm_prefill_chunk_size,
                self._base_lm_prefill_chunk_size,
                self._residual_lm_prefill_chunk_size,
            )

        locdit_path = (
            Path(locdit_model_path)
            if locdit_model_path is not None
            else mdir / f"locdit_p{patch_size}_c{patch_size}.mlpackage"
        )
        locdit_resolved_path = _resolve_model_path(locdit_path, use_compiled=True)
        self.locdit = CoreMLUnifiedCFM(
            str(locdit_resolved_path),
            in_channels=latent_dim,
            compute_units=locdit_compute_units,
        )

        vae_decoder_src = (
            Path(vae_decoder_model_path)
            if vae_decoder_model_path is not None
            else mdir / "audio_vae_decoder_lf4.mlpackage"
        )
        vae_decoder_path = _resolve_model_path(vae_decoder_src, use_compiled=True)
        self.vae_decoder = AudioVAEDecoder(
            str(vae_decoder_path),
            latent_frames=patch_size,
            latent_dim=latent_dim,
            upsample_factor=vae_decode_chunk_size,
            out_sample_rate=vae_out_sample_rate,
            compute_units=vae_decoder_compute_units,
        )

        fsq_src = (
            Path(fsq_model_path) if fsq_model_path is not None else mdir / "fsq_s4.mlpackage"
        )
        projections_src = (
            Path(projections_model_path)
            if projections_model_path is not None
            else mdir / "projections.mlpackage"
        )
        fsq_path = _resolve_model_path(fsq_src, use_compiled=True)
        projections_path = _resolve_model_path(projections_src, use_compiled=True)
        self.fsq_model = _load_coreml_model(
            fsq_path,
            compute_units=fsq_compute_units,
        )
        self.projections_model = _load_coreml_model(
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
        if self.debug:
            log.debug("VoxCPM2Generator.__init__ complete — all models loaded")

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
        return np.array(self._tokenizer.convert_tokens_to_ids(processed), dtype=np.int32)

    # ------------------------------------------------------------------ #
    # compatibility constructor for conversion/integration code
    # ------------------------------------------------------------------ #

    @classmethod
    def from_voxcpm(cls, model, model_dir: str, **kwargs) -> VoxCPM2Generator:
        """Create generator from a loaded VoxCPM model.

        Runtime code should prefer direct construction with
        ``embedding_safetensors_path`` so numpy inference does not require a
        loaded PyTorch model.
        """
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

    # ------------------------------------------------------------------ #
    # CoreML glue model helpers
    # ------------------------------------------------------------------ #

    def _fsq(self, x: np.ndarray) -> np.ndarray:
        """FSQ on NCHW tensor (1, C, 1, S) with chunked processing."""
        B, C, _, S = x.shape
        chunk = self.input_seq_length
        if S == chunk:
            return self.fsq_model.predict({"x": x.astype(np.float16)})["output"].astype(np.float32)
        # Pad to multiple of chunk
        pad = (chunk - S % chunk) % chunk
        if pad:
            x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, pad)))
        out_chunks = []
        for i in range(0, x.shape[-1], chunk):
            c = x[..., i:i + chunk].astype(np.float16)
            out_chunks.append(self.fsq_model.predict({"x": c})["output"])
        return np.concatenate(out_chunks, axis=-1)[..., :S].astype(np.float32)

    def _projections(self, lm_hidden: np.ndarray, residual_hidden: np.ndarray):
        """Returns (dit_hidden (1, 2*D, 1, 1), stop_flag (1, 2, 1, 1))."""
        out = self.projections_model.predict({
            "lm_hidden": lm_hidden.astype(np.float16),
            "residual_hidden": residual_hidden.astype(np.float16),
        })
        return out["dit_hidden"].astype(np.float32), out["stop_flag"].astype(np.float32)

    # ------------------------------------------------------------------ #
    # prefill helper (shared between generate and generate_streaming)
    # ------------------------------------------------------------------ #

    def _prefill(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        reference_wav_path: str = "",
        prompt_audio_feat: np.ndarray | None = None,
        reference_audio_feat: np.ndarray | None = None,
    ):
        """Build token/feature sequences and run LM prefill.

        Returns a dict with all the state needed to start the AR loop:
        ``lm_hidden``, ``residual_hidden``, ``prefix_feat_cond``,
        ``vae_decoder_reset`` flag.
        """
        t_prefill = time.perf_counter()
        if self.debug:
            log.debug("_prefill: start — target_text=%r, prompt_text=%r, "
                      "prompt_wav=%r, ref_wav=%r",
                      target_text[:80], prompt_text[:80],
                      prompt_wav_path, reference_wav_path)

        # 1. Build token + feature sequences. This mirrors upstream
        # VoxCPM2Model._generate mode dispatch:
        #   zero-shot, continuation, reference, reference+continuation.
        has_prompt_audio = prompt_audio_feat is not None or bool(prompt_wav_path)
        has_reference_audio = reference_audio_feat is not None or bool(reference_wav_path)
        has_prompt_text = bool(prompt_text)
        text = prompt_text + target_text if has_prompt_audio and has_prompt_text else target_text

        text_token = self._encode_text(text)
        text_token = np.concatenate([text_token, [self.audio_start_token]])
        text_length = len(text_token)
        if self.debug:
            log.debug("_prefill: text_length=%d", text_length)

        if prompt_audio_feat is not None:
            prompt_feat = np.asarray(prompt_audio_feat, dtype=np.float32)
        elif prompt_wav_path:
            if self.debug:
                log.debug("_prefill: encoding prompt wav...")
            prompt_feat = self.vae_encoder.encode_wav(prompt_wav_path, padding_mode="left")
        else:
            prompt_feat = np.empty((0, self.patch_size, self.latent_dim), dtype=np.float32)

        if reference_audio_feat is not None:
            ref_feat = np.asarray(reference_audio_feat, dtype=np.float32)
        elif reference_wav_path:
            if self.debug:
                log.debug("_prefill: encoding reference wav...")
            ref_feat = self.vae_encoder.encode_wav(reference_wav_path, padding_mode="right")
        else:
            ref_feat = np.empty((0, self.patch_size, self.latent_dim), dtype=np.float32)

        for name, feat in (("prompt_audio_feat", prompt_feat), ("reference_audio_feat", ref_feat)):
            if feat.ndim != 3 or feat.shape[1:] != (self.patch_size, self.latent_dim):
                raise ValueError(
                    f"{name} must have shape (T, {self.patch_size}, {self.latent_dim}), "
                    f"got {feat.shape}"
                )

        text_pad_feat = np.zeros(
            (text_length, self.patch_size, self.latent_dim), dtype=np.float32
        )

        if has_reference_audio:
            ref_tokens, ref_feats, ref_text_mask, ref_feat_mask = self._make_ref_prefix(ref_feat)
            if has_prompt_audio:
                prompt_audio_length = prompt_feat.shape[0]
                prompt_pad_token = np.zeros(prompt_audio_length, dtype=np.int32)
                text_token = np.concatenate([ref_tokens, text_token, prompt_pad_token])
                audio_feat = np.concatenate([ref_feats, text_pad_feat, prompt_feat], axis=0)
                text_mask_1d = np.concatenate([
                    ref_text_mask,
                    np.ones(text_length, dtype=np.float32),
                    np.zeros(prompt_audio_length, dtype=np.float32),
                ])
                feat_mask_1d = np.concatenate([
                    ref_feat_mask,
                    np.zeros(text_length, dtype=np.float32),
                    np.ones(prompt_audio_length, dtype=np.float32),
                ])
            else:
                text_token = np.concatenate([ref_tokens, text_token])
                audio_feat = np.concatenate([ref_feats, text_pad_feat], axis=0)
                text_mask_1d = np.concatenate([
                    ref_text_mask,
                    np.ones(text_length, dtype=np.float32),
                ])
                feat_mask_1d = np.concatenate([
                    ref_feat_mask,
                    np.zeros(text_length, dtype=np.float32),
                ])
        elif has_prompt_audio:
            audio_length = prompt_feat.shape[0]
            prompt_pad_token = np.zeros(audio_length, dtype=np.int32)
            text_token = np.concatenate([text_token, prompt_pad_token])
            audio_feat = np.concatenate([text_pad_feat, prompt_feat], axis=0)
            text_mask_1d = np.concatenate([
                np.ones(text_length, dtype=np.float32),
                np.zeros(audio_length, dtype=np.float32),
            ])
            feat_mask_1d = np.concatenate([
                np.zeros(text_length, dtype=np.float32),
                np.ones(audio_length, dtype=np.float32),
            ])
        else:
            audio_feat = text_pad_feat
            text_mask_1d = np.ones(text_length, dtype=np.float32)
            feat_mask_1d = np.zeros(text_length, dtype=np.float32)

        total_length = len(text_token)
        if self.debug:
            log.debug("_prefill: prompt_feat=%s ref_feat=%s total_length=%d",
                      prompt_feat.shape, ref_feat.shape, total_length)
            if not has_prompt_audio and not has_reference_audio:
                log.debug("_prefill: no prompt/reference wav, total_length=%d", total_length)

        text_mask = text_mask_1d.reshape(1, 1, 1, total_length)
        feat_mask = feat_mask_1d.reshape(1, 1, 1, total_length)

        # 2. Embeddings
        text_embed = self.embed_tokens[text_token] * self.scale_emb
        text_embed = text_embed.reshape(1, total_length, self.hidden_size).transpose(0, 2, 1)[:, :, None, :]

        if np.any(feat_mask_1d):
            feat_input = audio_feat.reshape(1, total_length, self.patch_size, self.latent_dim)
            feat_embed_flat = self.feat_encoder.encode_patches(feat_input)
            feat_embed = feat_embed_flat.transpose(0, 2, 1)[:, :, None, :]
        else:
            feat_embed = np.zeros((1, self.hidden_size, 1, total_length), dtype=np.float32)

        combined = text_mask * text_embed + feat_mask * feat_embed
        if self.debug:
            log.debug("_prefill: combined shape=%s, range=[%.4f, %.4f]",
                      combined.shape, float(combined.min()), float(combined.max()))

        prefix_feat_cond = audio_feat[-1].T.reshape(1, self.latent_dim, self.patch_size)
        if self.debug:
            log.debug("_prefill: prefix_feat_cond shape=%s range=[%.4f, %.4f]",
                      prefix_feat_cond.shape,
                      float(prefix_feat_cond.min()),
                      float(prefix_feat_cond.max()))

        # 3. Prefill base_lm
        if self.debug:
            log.debug(
                "_prefill: base_lm.forward (prefill_chunk=%s)...",
                self._base_lm_prefill_chunk_size,
            )
        t0 = time.perf_counter()
        enc_outputs, _ = self.base_lm.forward(
            combined,
            is_causal=True,
            reset_state=True,
            preferred_chunk_size=self._base_lm_prefill_chunk_size,
        )
        if self.debug:
            log.debug("_prefill: base_lm done in %.3fs, output shape=%s",
                      time.perf_counter() - t0, enc_outputs.shape)

        # 4. FSQ + mask blend
        fsq_out = self._fsq(enc_outputs)
        enc_outputs = fsq_out * feat_mask + enc_outputs * text_mask
        lm_hidden = enc_outputs[..., -1:]
        if self.debug:
            log.debug("_prefill: lm_hidden shape=%s", lm_hidden.shape)

        # 5. Prefill residual_lm
        feat_part = feat_mask * feat_embed
        residual_input = np.concatenate([enc_outputs, feat_part], axis=1)
        if self.debug:
            log.debug(
                "_prefill: residual_lm.forward (prefill_chunk=%s), input shape=%s...",
                self._residual_lm_prefill_chunk_size,
                residual_input.shape,
            )
        t0 = time.perf_counter()
        res_outputs, _ = self.residual_lm.forward(
            residual_input,
            is_causal=True,
            reset_state=True,
            preferred_chunk_size=self._residual_lm_prefill_chunk_size,
        )
        if self.debug:
            log.debug("_prefill: residual_lm done in %.3fs, output shape=%s",
                      time.perf_counter() - t0, res_outputs.shape)
        residual_hidden = res_outputs[..., -1:]

        dt = time.perf_counter() - t_prefill
        if self.debug:
            log.debug("_prefill: complete in %.3fs", dt)

        return {
            "lm_hidden": lm_hidden,
            "residual_hidden": residual_hidden,
            "prefix_feat_cond": prefix_feat_cond,
        }

    def _make_ref_prefix(self, ref_feat: np.ndarray):
        """Build upstream-compatible [ref_start ref_audio ref_end] prefix."""
        ref_len = ref_feat.shape[0]
        z1 = np.zeros((1, self.patch_size, self.latent_dim), dtype=np.float32)
        tokens = np.concatenate([
            np.array([self.ref_audio_start_token], dtype=np.int32),
            np.zeros(ref_len, dtype=np.int32),
            np.array([self.ref_audio_end_token], dtype=np.int32),
        ])
        feats = np.concatenate([z1, ref_feat, z1], axis=0)
        text_mask = np.concatenate([
            np.ones(1, dtype=np.float32),
            np.zeros(ref_len, dtype=np.float32),
            np.ones(1, dtype=np.float32),
        ])
        feat_mask = np.concatenate([
            np.zeros(1, dtype=np.float32),
            np.ones(ref_len, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        ])
        return tokens, feats, text_mask, feat_mask

    # ------------------------------------------------------------------ #
    # LM idle swap
    # ------------------------------------------------------------------ #

    def _swap_to_idle(self) -> None:
        """Reload the prefill function and unload the decode function.

        Called after generation completes so the next request gets a fast
        prefill without a cold function load. Only has effect when
        ``lm_unload_inactive_functions`` and ``lm_idle_prefill_chunk_size``
        are configured.
        """
        for name, lm in [("base_lm", self.base_lm), ("residual_lm", self.residual_lm)]:
            if not hasattr(lm, "idle_prefill_chunk_size"):
                if self.debug:
                    log.debug("_swap_to_idle: %s has no idle_prefill_chunk_size, skip", name)
                continue
            if not getattr(lm, "unload_inactive_functions", False):
                if self.debug:
                    log.debug("_swap_to_idle: %s unload_inactive=False, skip", name)
                continue
            idle = lm.idle_prefill_chunk_size
            if idle is None:
                if self.debug:
                    log.debug("_swap_to_idle: %s idle_prefill_chunk_size=None, skip", name)
                continue
            if self.debug:
                log.debug("_swap_to_idle: %s unloading s%d, preloading s%d",
                          name, lm.chunk_size, idle)
            # Unload the decode function (chunk_size = default, e.g. 1)
            lm._unload_function(
                lm.chunk_size,
                event_name=f"idle/unload_function_s{lm.chunk_size}",
            )
            lm.model = None
            # Preload the prefill function
            lm._model_for_chunk_size(idle, profile_prefix="idle")
            if self.debug:
                log.debug("_swap_to_idle: %s done", name)

    def _swap_to_decode(self) -> None:
        """Load the decode function immediately after prefill.

        CoreML's multifunction MLState path is sensitive to when function
        handles are loaded.  For hot-swap mode, move both LMs to their decode
        function before the first autoregressive iteration instead of lazily
        loading it after the first streamed audio chunk.
        """
        for name, lm in [("base_lm", self.base_lm), ("residual_lm", self.residual_lm)]:
            if not hasattr(lm, "chunk_size"):
                if self.debug:
                    log.debug("_swap_to_decode: %s has no chunk_size, skip", name)
                continue
            if not getattr(lm, "unload_inactive_functions", False):
                if self.debug:
                    log.debug("_swap_to_decode: %s unload_inactive=False, skip", name)
                continue
            if self.debug:
                log.debug("_swap_to_decode: %s loading s%d", name, lm.chunk_size)
            lm._model_for_chunk_size(lm.chunk_size, profile_prefix="decode")
            if self.debug:
                log.debug("_swap_to_decode: %s done", name)

    # ------------------------------------------------------------------ #
    # main generation
    # ------------------------------------------------------------------ #

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
        seed: int = 1234,
    ) -> np.ndarray:
        """Generate audio from text + optional reference voice.

        Returns a 1-D float32 waveform at ``self.vae_out_sample_rate`` Hz.
        """
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
        lm_hidden = state["lm_hidden"]
        residual_hidden = state["residual_hidden"]
        prefix_feat_cond = state["prefix_feat_cond"]

        # Autoregressive loop
        pred_feat_seq = []

        try:
            for i in range(max_len):
                if self.debug:
                    log.debug("generate: AR step %d/%d", i, max_len)
                    t_step = time.perf_counter()

                dit_hidden, stop_flag = self._projections(lm_hidden, residual_hidden)
                if self.debug:
                    log.debug("generate[%d]: projections done, dit_hidden shape=%s, stop_flag=%s",
                              i, dit_hidden.shape, stop_flag.reshape(-1).tolist())

                pred_feat = self.locdit.predict_numpy(
                    mu=dit_hidden.reshape(1, -1),
                    n_timesteps=inference_timesteps,
                    patch_size=self.patch_size,
                    cond=prefix_feat_cond,
                    cfg_value=cfg_value,
                    rng=rng,
                )
                if self.debug:
                    log.debug("generate[%d]: locdit done, pred_feat shape=%s range=[%.4f, %.4f]",
                              i, pred_feat.shape, float(pred_feat.min()), float(pred_feat.max()))

                pf = pred_feat.transpose(0, 2, 1)
                pf = pf.reshape(1, 1, self.patch_size, self.latent_dim)
                curr_embed = self.feat_encoder.encode_patches(pf)
                curr_embed_nchw = curr_embed.transpose(0, 2, 1)[:, :, None, :]

                pred_feat_seq.append(pred_feat)
                prefix_feat_cond = pred_feat

                stop = int(np.argmax(stop_flag.reshape(-1)))
                if i > min_len and stop == 1:
                    if self.debug:
                        log.debug("generate[%d]: stop token detected, breaking", i)
                    break

                lm_hidden = self.base_lm.forward_step(curr_embed_nchw, None)
                lm_hidden = self._fsq(lm_hidden)

                residual_input = np.concatenate([lm_hidden, curr_embed_nchw], axis=1)
                residual_hidden = self.residual_lm.forward_step(residual_input, None)
                if self.debug:
                    log.debug("generate[%d]: step done in %.3fs",
                              i, time.perf_counter() - t_step)
        finally:
            if self.debug:
                log.debug("generate: AR loop done, %d patches, swapping to idle", len(pred_feat_seq))
            self._swap_to_idle()

        # Decode audio from predicted features
        all_feats = np.concatenate(pred_feat_seq, axis=-1)
        latent = all_feats[0]

        self.vae_decoder.reset()
        upsample = self.vae_decode_chunk_size
        total_P = latent.shape[-1]
        audio_parts = []

        for start in range(0, total_P, self.patch_size):
            chunk = latent[:, start:start + self.patch_size].reshape(
                1, self.latent_dim, self.patch_size
            )
            audio_parts.append(self.vae_decoder.decode_chunk(chunk).reshape(-1))

        return np.concatenate(audio_parts)[:total_P * upsample]

    # ------------------------------------------------------------------ #
    # streaming generation
    # ------------------------------------------------------------------ #

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
        seed: int = 1234,
    ) -> Generator[np.ndarray, None, None]:
        """Like :meth:`generate`, but yields audio chunks per AR step.

        Each autoregressive step produces one patch of latent features,
        which is immediately VAE-decoded into audio and yielded. This
        matches the upstream ``VoxCPM2Model._generate(streaming=True)``
        pattern.

        Yields:
            1-D float32 audio chunks at ``self.vae_out_sample_rate`` Hz.
        """
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
        lm_hidden = state["lm_hidden"]
        residual_hidden = state["residual_hidden"]
        prefix_feat_cond = state["prefix_feat_cond"]

        self.vae_decoder.reset()
        if self.debug:
            log.debug("generate_streaming: vae_decoder reset, starting AR loop")

        try:
            for i in range(max_len):
                if self.debug:
                    log.debug("stream[%d]: AR step start", i)
                    t_step = time.perf_counter()

                dit_hidden, stop_flag = self._projections(lm_hidden, residual_hidden)
                if self.debug:
                    log.debug("stream[%d]: projections done, dit_hidden shape=%s, "
                              "stop_flag=%s",
                              i, dit_hidden.shape, stop_flag.reshape(-1).tolist())

                pred_feat = self.locdit.predict_numpy(
                    mu=dit_hidden.reshape(1, -1),
                    n_timesteps=inference_timesteps,
                    patch_size=self.patch_size,
                    cond=prefix_feat_cond,
                    cfg_value=cfg_value,
                    rng=rng,
                )
                if self.debug:
                    log.debug("stream[%d]: locdit done, pred_feat shape=%s range=[%.4f, %.4f]",
                              i, pred_feat.shape, float(pred_feat.min()), float(pred_feat.max()))

                # VAE decode this patch immediately
                if self.debug:
                    has_nan = bool(np.isnan(pred_feat).any())
                    has_inf = bool(np.isinf(pred_feat).any())
                    log.debug("stream[%d]: vae_decoder.decode_chunk "
                              "pred_feat=%s nan=%s inf=%s range=[%.4f, %.4f]",
                              i, pred_feat.shape, has_nan, has_inf,
                              float(pred_feat.min()), float(pred_feat.max()))
                    if hasattr(self.vae_decoder, '_caches'):
                        for ci, c in enumerate(self.vae_decoder._caches):
                            log.debug("stream[%d]: vae cache_%d shape=%s "
                                      "range=[%.4f, %.4f] nan=%s",
                                      i, ci, c.shape, float(c.min()),
                                      float(c.max()), bool(np.isnan(c).any()))
                    sys.stdout.flush(); sys.stderr.flush()
                audio_chunk = self.vae_decoder.decode_chunk(pred_feat).reshape(-1)
                if self.debug:
                    log.debug("stream[%d]: audio_chunk len=%d range=[%.4f, %.4f]",
                              i, len(audio_chunk),
                              float(audio_chunk.min()), float(audio_chunk.max()))
                yield audio_chunk

                pf = pred_feat.transpose(0, 2, 1)
                pf = pf.reshape(1, 1, self.patch_size, self.latent_dim)
                curr_embed = self.feat_encoder.encode_patches(pf)
                curr_embed_nchw = curr_embed.transpose(0, 2, 1)[:, :, None, :]

                prefix_feat_cond = pred_feat

                stop = int(np.argmax(stop_flag.reshape(-1)))
                if i > min_len and stop == 1:
                    if self.debug:
                        log.debug("stream[%d]: stop token detected, breaking", i)
                    break

                if self.debug:
                    log.debug("stream[%d]: base_lm.forward_step...", i)
                lm_hidden = self.base_lm.forward_step(curr_embed_nchw, None)
                lm_hidden = self._fsq(lm_hidden)

                if self.debug:
                    log.debug("stream[%d]: residual_lm.forward_step...", i)
                residual_input = np.concatenate([lm_hidden, curr_embed_nchw], axis=1)
                residual_hidden = self.residual_lm.forward_step(residual_input, None)
                if self.debug:
                    log.debug("stream[%d]: step done in %.3fs",
                              i, time.perf_counter() - t_step)
        except GeneratorExit:
            if self.debug:
                log.debug("generate_streaming: GeneratorExit caught")
        finally:
            if self.debug:
                log.debug("generate_streaming: AR loop finished, swapping to idle")
            self._swap_to_idle()

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #

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
