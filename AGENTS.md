# Agent Instructions

This repository contains the source code for the **VoxCPMANE** package, the canonical VoxCPM2 TTS runtime and HTTP server using CoreML and Apple Neural Engine.

## Project Overview

*   **Core Purpose**: A pip-installable package (`voxcpmane`) providing:
    1. Pure numpy/CoreML runtime wrappers for VoxCPM2 model components (LM, LocDiT, FeatEncoder, AudioVAE).
    2. A `VoxCPM2Generator` class orchestrating the full TTS pipeline.
    3. A FastAPI HTTP server (`voxcpmane.server`) serving TTS via OpenAI-compatible endpoints.
*   **Key Dependencies**:
    *   `coremltools`: For running the ML models on macOS Apple Silicon.
    *   `sounddevice`: For audio playback.
    *   `fastapi`, `uvicorn`: Web server.
    *   `transformers`: Tokenization.
*   **Structure**:
    *   `src/voxcpmane/`: Source code.
    *   `src/voxcpmane/frontend/`: Static frontend assets.

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

## Architecture Note

This package is the **canonical home** for VoxCPM2 runtime modules. The parent
repository (`qeml`) imports from `voxcpmane` via thin re-export shims in
`src/qeml/voxcpm2/`. Do not duplicate runtime code back into `qeml`.
