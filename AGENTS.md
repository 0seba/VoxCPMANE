# Agent Instructions

This repository contains the source code for the **VoxCPMANE** server, a text-to-speech system using CoreML and Apple Neural Engine. This file provides critical instructions for AI agents working on this codebase.

## Project Overview

*   **Core Purpose**: A FastAPI server (`src/voxcpmane/server.py`) that serves a VoxCPM TTS model.
*   **Key Dependencies**:
    *   `coremltools`: For running the ML models on macOS Apple Silicon.
    *   `sounddevice`: For audio playback.
    *   `fastapi`, `uvicorn`: Web server.
    *   `transformers`: Tokenization and model utilities.
*   **Structure**:
    *   `src/voxcpmane/`: Source code.
    *   `src/voxcpmane/frontend/`: Static frontend assets (served by FastAPI).
    *   `docs/`: Documentation.

## ⚠️ Environment & Testing Limitations

**CRITICAL**: This project relies on `coremltools` (requires macOS) and `sounddevice` (requires PortAudio/system audio). Most CI/CD environments and Agent Sandboxes (running Linux) **CANNOT** run the server or the model directly.

### Mandatory Mocking Strategy

You **MUST** use `unittest.mock` to mock hardware-dependent modules when writing tests or verifying code imports.

The following modules **must** be mocked in any test script you write:
1.  `coremltools`
2.  `sounddevice`
3.  `soundfile` (if `libsndfile` is missing)
4.  `soxr`
5.  `huggingface_hub.snapshot_download` (to prevent large model downloads during tests)

#### Example Mock Setup

Use this pattern at the top of your test scripts *before* importing any project modules:

```python
import sys
from unittest.mock import MagicMock
import types

# Helper for module mocking
def create_mock_module(name):
    m = types.ModuleType(name)
    m.__spec__ = MagicMock()
    m.__spec__.origin = "mock"
    m.__file__ = "mock"
    return m

# Mock hardware dependencies
sys.modules["sounddevice"] = create_mock_module("sounddevice")
sys.modules["soundfile"] = create_mock_module("soundfile")
sys.modules["soxr"] = create_mock_module("soxr")

coremltools = create_mock_module("coremltools")
coremltools.models = MagicMock()
sys.modules["coremltools"] = coremltools
sys.modules["coremltools.models"] = coremltools.models

# Mock HuggingFace downloads
import huggingface_hub
huggingface_hub.snapshot_download = MagicMock(return_value="/tmp/mock_model_path")

# Mock numpy.load to handle fake model files
import numpy as np
original_load = np.load
def mock_load(file, *args, **kwargs):
    if str(file).endswith(".npy"):
        return np.zeros((1, 1))
    return original_load(file, *args, **kwargs)
np.load = mock_load

# Mock os path checks for model files
import os
original_exists = os.path.exists
def mock_exists(path):
    if "/tmp/mock_model_path" in str(path):
        return True
    return original_exists(path)
os.path.exists = mock_exists
```

## Workflow & Expectations

1.  **Scope of Work**: You are expected to implement new features, fix bugs, and add tests.
2.  **Verification**: Because you cannot run the server, you must write **unit tests** using the mocking strategy above to verify your logic.
3.  **Dependency Management**:
    *   This project uses `uv`.
    *   Install dependencies: `uv sync` or `uv pip install -e .`
    *   Run scripts: `uv run python my_script.py`

## Code Style

*   **Formatter**: You **MUST** use `black` for code formatting.
*   **Run Black**: `uv run black .` before submitting any changes.

## Documentation

*   **API Changes**: The API documentation is located in `docs/API.md`. If you modify endpoints or request/response structures, you **MUST** update this file.
*   **README**: If you add major features, update `README.md` to reflect them (linking to `docs/API.md` for details).

## Directory Structure

*   Do not edit build artifacts or files in `dist/` or `build/`.
*   Frontend files are in `src/voxcpmane/frontend/`. If you modify `index.html`, ensure the server serves it correctly.
