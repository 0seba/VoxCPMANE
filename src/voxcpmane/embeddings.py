"""Utilities for loading the LM token embedding table."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_EMBEDDING_KEYS = (
    "tts_model.base_lm.embed_tokens.weight",
    "base_lm.embed_tokens.weight",
    "model.tts_model.base_lm.embed_tokens.weight",
    "lm.embed_tokens.weight",
    "embed_tokens.weight",
)


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

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "loading embeddings from .safetensors requires the optional "
            "development dependency: install voxcpmane2[development] or provide "
            "embed_tokens.npy in the model directory"
        ) from exc

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
                    return tensors.get_tensor(selected).astype(np.float16)
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
                    return tensors.get_tensor(selected).astype(np.float16)

    preview = ", ".join(sorted(set(available))[:12])
    raise KeyError(
        f"could not find embedding tensor in safetensors files under {path}; "
        f"looked for {', '.join(keys)}. Available keys include: {preview}"
    )


def _resolve_hf_cache_dir(
    repo_id: str = "openbmb/VoxCPM2",
) -> Path | None:
    """Return the latest local HuggingFace snapshot for *repo_id*, if cached."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return None

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return None

    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            # Pick the most recently modified revision.
            revisions = sorted(
                repo.revisions,
                key=lambda r: r.last_modified,
                reverse=True,
            )
            if revisions:
                return revisions[0].snapshot_path
    return None


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
            return np.load(str(search_path)).astype(np.float16)

        if search_path.is_dir():
            npy = search_path / "embed_tokens.npy"
            if npy.exists():
                return np.load(str(npy)).astype(np.float16)

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
