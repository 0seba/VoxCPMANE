from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import subprocess
import sys

import numpy as np
import pytest

from voxcpmane import server
from voxcpmane.generator import VoxCPM2Generator
from voxcpmane.lm import CoreMLMiniCPMLM


@pytest.fixture
def startup(monkeypatch, tmp_path):
    monkeypatch.delenv("VOXCPMANE2_M5_MAX_MODEL_DIR", raising=False)
    main = tmp_path / "main"
    debug = tmp_path / "debug"
    (main / "caches").mkdir(parents=True)
    for root in (main, debug):
        for name in (*server.M5_MAX_LM_PACKAGES, *server.SPLIT_BASE_LM_COMPILED_PARTS):
            package = root / name
            package.mkdir(parents=True)
            (package / "metadata.json").write_text("[]")
            (package / "model.mil").write_text("")
    download = Mock(side_effect=lambda **kw: str(
        debug if kw["repo_id"] == server.SPLIT_BASE_LM_REPO_ID else main
    ))
    factory = Mock()
    monkeypatch.setattr(server, "snapshot_download", download)
    monkeypatch.setattr(server, "VoxCPM2Generator", factory)
    monkeypatch.setattr(server, "suggest_m5_max_compat", Mock())
    monkeypatch.setattr(server, "generator", None)
    monkeypatch.setattr(server, "CUSTOM_VOICE_CACHE_DIR", str(tmp_path / "custom"))
    return main, debug, download, factory


@pytest.mark.parametrize("profile", ["default", "split", "m5"])
def test_download_selection_with_old_models_cached(startup, profile):
    main, debug, download, factory = startup
    server.load_model(split_base_lm=profile == "split", m5_max_compat=profile == "m5")
    patterns = download.call_args_list[0].kwargs["allow_patterns"]
    assert "config.json" in patterns
    assert ("base_lm_multifunction.mlmodelc/**" in patterns) == (profile == "default")
    assert ("residual_lm_fused_multifunction.mlmodelc/**" in patterns) == (profile != "m5")
    debug_calls = [c.kwargs for c in download.call_args_list
                   if c.kwargs["repo_id"] == server.SPLIT_BASE_LM_REPO_ID]
    kwargs = factory.call_args.kwargs
    if profile == "m5":
        assert debug_calls == [{
            "repo_id": server.SPLIT_BASE_LM_REPO_ID,
            "revision": server.M5_MAX_LM_REVISION,
            "allow_patterns": ["config.json", *server.compiled_package_patterns(server.M5_MAX_LM_PACKAGES)],
        }]
        assert kwargs["base_lm_model_path"] == debug / server.M5_MAX_LM_PACKAGES[0]
        assert kwargs["residual_lm_model_path"] == debug / server.M5_MAX_LM_PACKAGES[1]
        assert kwargs["lm_cache_profile"] == server.M5_MAX_CACHE_PROFILE
    elif profile == "split":
        assert len(kwargs["base_lm_split_model_paths"]) == 2
        assert "base_lm_model_path" not in kwargs
    else:
        assert not debug_calls
        assert kwargs["base_lm_model_path"].parent == main


@pytest.mark.parametrize("conflict", [
    {"split_base_lm": True}, {"base_lm_path": ["local.mlmodelc"]},
    {"residual_lm_path": "local.mlmodelc"},
])
def test_conflicts_fail_before_download(startup, conflict):
    with pytest.raises(ValueError, match="cannot be combined"):
        server.load_model(m5_max_compat=True, **conflict)
    startup[2].assert_not_called()


def test_local_components_use_corrected_remote_lms(startup):
    main, debug, download, factory = startup
    server.load_model(model_dir=str(main), m5_max_compat=True)
    assert download.call_args_list[0].kwargs["revision"] == server.M5_MAX_LM_REVISION
    assert factory.call_args.args[0] == main
    assert factory.call_args.kwargs["base_lm_model_path"].parent == debug


def test_incomplete_corrected_model_does_not_fall_back(startup):
    (startup[1] / server.M5_MAX_LM_PACKAGES[0] / "model.mil").unlink()
    with pytest.raises(FileNotFoundError):
        server.load_model(m5_max_compat=True)
    startup[3].assert_not_called()


def test_local_compatibility_override_skips_debug_download(startup, monkeypatch):
    main, debug, download, factory = startup
    monkeypatch.setenv("VOXCPMANE2_M5_MAX_MODEL_DIR", str(debug))
    server.load_model(m5_max_compat=True)
    assert all(c.kwargs["repo_id"] != server.SPLIT_BASE_LM_REPO_ID
               for c in download.call_args_list)
    kwargs = factory.call_args.kwargs
    assert kwargs["base_lm_model_path"] == debug / server.M5_MAX_LM_PACKAGES[0]
    assert kwargs["residual_lm_model_path"] == debug / server.M5_MAX_LM_PACKAGES[1]
    assert kwargs["lm_cache_profile"] == server.M5_MAX_CACHE_PROFILE


def test_missing_local_compatibility_override_fails_without_debug_download(startup, monkeypatch):
    monkeypatch.setenv("VOXCPMANE2_M5_MAX_MODEL_DIR", str(startup[1] / "missing"))
    with pytest.raises(FileNotFoundError):
        server.load_model(m5_max_compat=True)
    assert all(c.kwargs["repo_id"] != server.SPLIT_BASE_LM_REPO_ID
               for c in startup[2].call_args_list)
    startup[3].assert_not_called()


def test_cli_forwards_compatibility_flag(monkeypatch, tmp_path):
    load = Mock()
    run = Mock()
    monkeypatch.setattr(sys, "argv", [
        "voxcpmane2-server", "--m5-max-compat", "--cache-dir", str(tmp_path),
    ])
    monkeypatch.setattr(server, "load_model", load)
    monkeypatch.setattr(server.uvicorn, "run", run)
    monkeypatch.setattr(server.VOICE_STORE, "available", lambda: [])
    monkeypatch.setattr(server, "CUSTOM_VOICE_CACHE_DIR", "")
    monkeypatch.setattr(server.metrics, "LIVE_RTF_METRICS", "off")
    server.main()
    assert load.call_args.kwargs["m5_max_compat"] is True
    assert load.call_args.kwargs["lm_mode"] == "single-length"
    run.assert_called_once()


def test_non_macos_skips_hardware_detection(monkeypatch):
    monkeypatch.setattr(server.platform, "system", lambda: "Linux")
    run = Mock()
    monkeypatch.setattr(server.subprocess, "run", run)
    server.suggest_m5_max_compat(False)
    run.assert_not_called()


@pytest.mark.parametrize("chip,enabled,expected", [
    ("Apple M5 Max\n", False, True), (" Apple  M5 MAX ", False, True),
    ("Apple M5", False, False), ("Apple M3 Max", False, False),
    ("Apple M5 Max", True, False),
])
def test_hardware_advisory(monkeypatch, capsys, chip, enabled, expected):
    monkeypatch.setattr(server.platform, "system", lambda: "Darwin")
    run = Mock(return_value=SimpleNamespace(stdout=chip))
    monkeypatch.setattr(server.subprocess, "run", run)
    server.suggest_m5_max_compat(enabled)
    assert ("--m5-max-compat" in capsys.readouterr().out) == expected
    if enabled:
        run.assert_not_called()


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("sysctl", 2)])
def test_hardware_detection_failure_is_nonfatal(monkeypatch, error):
    monkeypatch.setattr(server.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(server.subprocess, "run", Mock(side_effect=error))
    server.suggest_m5_max_compat(False)


def test_cache_namespace_and_profile_validation(monkeypatch, tmp_path):
    state = {"key_cache": np.zeros((1, 2, 3, 4), np.float16)}
    lm = Mock(is_chain=False, state_shapes=[("key_cache", (1, 2, 16, 4))])
    lm.snapshot_state_prefix.return_value = state
    gen = object.__new__(VoxCPM2Generator)
    gen.base_lm = gen.residual_lm = lm
    gen.lm_cache_profile = server.M5_MAX_CACHE_PROFILE
    monkeypatch.setattr(server, "generator", gen)
    monkeypatch.setattr(server, "VOICE_CACHE_DIRS", [])
    monkeypatch.setattr(server, "CUSTOM_VOICE_CACHE_DIR", str(tmp_path))
    _, filename = server.VOICE_STORE.lm_prefix_paths("voice")
    assert server.M5_MAX_CACHE_PROFILE in filename
    path = Path(filename)
    gen._save_lm_prefix_cache(path, 3)
    assert gen._restore_lm_prefix_cache(path, 3)
    lm.restore_state_prefix.reset_mock()
    gen.lm_cache_profile = None
    assert not gen._restore_lm_prefix_cache(path, 3)
    lm.restore_state_prefix.assert_not_called()
    gen._save_lm_prefix_cache(path, 3)
    gen.lm_cache_profile = server.M5_MAX_CACHE_PROFILE
    assert not gen._restore_lm_prefix_cache(path, 3)


@pytest.mark.parametrize("mode", ["single-length", "preload", "always-loaded", "hot-swap"])
def test_lm_lifecycle_modes(monkeypatch, mode):
    from voxcpmane import lm as runtime
    sizes = (1, 8, 16, 32, 64, 128, 256)
    def description(size, name):
        return {"name": name, "inputSchema": [
            {"name": "inputs_embeds", "shape": f"[1, 2048, 1, {size}]"},
            {"name": "position_id", "shape": "[1]"},
            {"name": "position_index_seed", "shape": f"[{size}]"},
        ]}
    metadata = description(1, "main")
    metadata["functions"] = [description(1, "main"), *[
        description(s, f"length_{s}") for s in sizes
    ]]
    monkeypatch.setattr(runtime, "load_compiled_metadata_entry", lambda p: metadata)
    def load(*args, **kwargs):
        handle = Mock()
        handle.predict.side_effect = lambda inputs, state: {
            "hidden_states": inputs["inputs_embeds"] + 1
        }
        return handle
    monkeypatch.setattr(runtime, "load_coreml_model", load)
    mode_kwargs = server.lm_mode_kwargs(mode, 16, residual_lm_path="multifunction")
    options = {
        "unload_inactive_functions": mode_kwargs["lm_unload_inactive_functions"],
        "restrict_to_preload": mode_kwargs["lm_restrict_to_preload"],
        "preload_chunk_sizes": mode_kwargs["lm_preload_chunk_sizes"],
        "idle_prefill_chunk_size": mode_kwargs.get("lm_idle_prefill_chunk_size"),
        "keep_default_function_loaded": mode_kwargs.get("lm_keep_decode_function_loaded", False),
    }
    gen = object.__new__(VoxCPM2Generator)
    gen.base_lm = CoreMLMiniCPMLM(Path("base.mlmodelc"), **options)
    gen.residual_lm = CoreMLMiniCPMLM(Path("residual.mlmodelc"), **options)
    for _ in range(2):
        for lm in (gen.base_lm, gen.residual_lm):
            lm.forward(np.zeros((1, 2048, 1, 19), np.float32), preferred_chunk_size=16)
        gen._swap_to_decode()
        expected = {16} if mode == "single-length" else {1, 16} if mode == "always-loaded" else {1}
        for lm in (gen.base_lm, gen.residual_lm):
            assert set(lm._models_by_chunk_size) == expected
            lm.forward_step(np.zeros((1, 2048, 1, 1), np.float32))
            assert lm.current_position == 20
        gen._swap_to_idle()
        expected = {16} if mode in {"single-length", "hot-swap"} else {1, 16}
        for lm in (gen.base_lm, gen.residual_lm):
            assert set(lm._models_by_chunk_size) == expected
