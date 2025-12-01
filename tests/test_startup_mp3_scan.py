import sys
import os
import unittest
from unittest.mock import MagicMock, patch, mock_open
import types
import numpy as np


# --- MOCKING SETUP START ---
def create_mock_module(name):
    m = types.ModuleType(name)
    m.__spec__ = MagicMock()
    m.__spec__.origin = "mock"
    m.__file__ = "mock"
    return m


sys.modules["sounddevice"] = create_mock_module("sounddevice")
sys.modules["soundfile"] = create_mock_module("soundfile")
sys.modules["soxr"] = create_mock_module("soxr")
# Mock pydub
pydub = create_mock_module("pydub")
pydub.AudioSegment = MagicMock()
sys.modules["pydub"] = pydub

coremltools = create_mock_module("coremltools")
coremltools.models = MagicMock()
coremltools.models.CompiledMLModel = MagicMock()
coremltools.ComputeUnit = MagicMock()
sys.modules["coremltools"] = coremltools
sys.modules["coremltools.models"] = coremltools.models

import huggingface_hub

huggingface_hub.snapshot_download = MagicMock(return_value="/tmp/mock_model_path")

# Mock numpy.load
original_load = np.load


def mock_load(file, *args, **kwargs):
    if isinstance(file, str) and str(file).endswith(".npy"):
        return np.zeros((1, 1))
    return original_load(file, *args, **kwargs)


np.load = mock_load

original_exists = os.path.exists


def mock_exists(path):
    if "/tmp/mock_model_path" in str(path):
        return True
    return original_exists(path)


os.path.exists = mock_exists
# --- MOCKING SETUP END ---

# Mock VoxCPMANE before import
with patch("voxcpmane.voxcpm.VoxCPMANE") as MockVox:
    from voxcpmane.server import scan_and_compile_mp3_cache, model


class TestStartupScan(unittest.TestCase):
    @patch("voxcpmane.server.CUSTOM_VOICE_CACHE_DIR", "/tmp/custom_cache")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("builtins.open", new_callable=mock_open, read_data="transcript")
    @patch("voxcpmane.server.AudioSegment")
    @patch("voxcpmane.server.model")  # Mock the global model instance
    @patch("voxcpmane.server.PYDUB_AVAILABLE", True)
    def test_scan_and_compile(
        self, mock_model, mock_audio_segment, mock_file, mock_listdir, mock_exists
    ):
        # Setup
        def side_effect_exists(path):
            if path == "/tmp/custom_cache":
                return True
            return False

        mock_exists.side_effect = side_effect_exists
        mock_listdir.return_value = [
            "voice1.mp3",
            "voice1.txt",
            "voice2.mp3",
            "voice3.txt",
            "voice4.npy",
        ]

        # Mock AudioSegment
        mock_segment = MagicMock()
        mock_audio_segment.from_mp3.return_value = mock_segment

        # Run
        scan_and_compile_mp3_cache()

        # Verifications

        # voice1: Both mp3 and txt exist, npy missing. Should trigger creation.
        mock_audio_segment.from_mp3.assert_called_with("/tmp/custom_cache/voice1.mp3")
        mock_segment.export.assert_called()
        mock_model.create_custom_voice.assert_called_with(
            voice_name="voice1",
            prompt_wav_path=mock_segment.export.call_args[0][0],  # temp file path
            prompt_text="transcript",
            cache_dir="/tmp/custom_cache",
        )

        # voice2: mp3 only. Should warn.
        # voice3: txt only. Should warn.
        # voice4: npy exists. Should ignore.

        # Verify create_custom_voice was called ONLY ONCE (for voice1)
        self.assertEqual(mock_model.create_custom_voice.call_count, 1)


if __name__ == "__main__":
    unittest.main()
