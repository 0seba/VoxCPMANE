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
    from voxcpmane.server import scan_and_compile_audio_cache, model


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

        # Scenario:
        # voice1: mp3 + txt -> create
        # voice2: wav + txt -> create
        # voice3: flac only -> fail/warn
        # voice4: txt only -> fail/warn
        # voice5: npy exists -> ignore
        mock_listdir.return_value = [
            "voice1.mp3",
            "voice1.txt",
            "voice2.wav",
            "voice2.txt",
            "voice3.flac",
            "voice4.txt",
            "voice5.npy",
        ]

        # Mock AudioSegment
        mock_segment = MagicMock()
        mock_audio_segment.from_file.return_value = mock_segment

        # Run
        scan_and_compile_audio_cache()

        # Verifications

        # voice1 (mp3)
        mock_audio_segment.from_file.assert_any_call("/tmp/custom_cache/voice1.mp3")

        # voice2 (wav)
        mock_audio_segment.from_file.assert_any_call("/tmp/custom_cache/voice2.wav")

        # Verify create_custom_voice was called TWICE (voice1 and voice2)
        self.assertEqual(mock_model.create_custom_voice.call_count, 2)

        # Check calls args
        calls = mock_model.create_custom_voice.call_args_list
        voice_names = {call[1]["voice_name"] for call in calls}
        self.assertEqual(voice_names, {"voice1", "voice2"})


if __name__ == "__main__":
    unittest.main()
