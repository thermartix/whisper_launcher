"""Regression checks for mixed-language setup, caching, and device fallback.

Run with .whisper-runner/venv-fast/Scripts/python.exe -m unittest discover -s tests -v.
"""

import importlib.util
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch


spec = importlib.util.spec_from_file_location("transcribe", Path(__file__).resolve().parents[1] / "transcribe.py")
transcribe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcribe)
sys.modules["transcribe"] = transcribe


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.snapshot = Path(self.temporary.name)
        for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
            (self.snapshot / name).write_text("cached", encoding="utf-8")

    @patch("huggingface_hub.snapshot_download")
    def test_complete_cache_never_contacts_hub(self, download):
        download.return_value = str(self.snapshot)
        self.assertEqual(transcribe.prepare_mixed_model("medium"), str(self.snapshot))
        download.assert_called_once()
        self.assertTrue(download.call_args.kwargs["local_files_only"])

    @patch("huggingface_hub.snapshot_download")
    def test_partial_snapshot_is_downloaded_with_visible_progress(self, download):
        (self.snapshot / "model.bin").write_bytes(b"")

        def fetch(**options):
            if not options.get("local_files_only"):
                (self.snapshot / "model.bin").write_bytes(b"downloaded")
                output = io.StringIO()
                with options["tqdm_class"](total=10, unit="B", file=output, position=2) as progress:
                    progress.update(10)
                    progress.refresh()
                self.assertIn("100%", output.getvalue())
                self.assertNotIn("\x1b", output.getvalue())
            return str(self.snapshot)

        download.side_effect = fetch
        self.assertEqual(transcribe.prepare_mixed_model("medium"), str(self.snapshot))
        self.assertEqual(download.call_count, 2)

    @patch("huggingface_hub.snapshot_download")
    def test_first_download_and_network_failure_have_actionable_error(self, download):
        from huggingface_hub.errors import LocalEntryNotFoundError

        download.side_effect = [LocalEntryNotFoundError("not cached"), OSError("connection interrupted")]
        with self.assertRaisesRegex(SystemExit, "internet connection and free disk space"):
            transcribe.prepare_mixed_model("medium")

    @patch("huggingface_hub.snapshot_download")
    def test_incomplete_download_is_not_loaded(self, download):
        (self.snapshot / "vocabulary.txt").unlink()
        download.return_value = str(self.snapshot)
        with self.assertRaisesRegex(SystemExit, "model cache is incomplete"):
            transcribe.prepare_mixed_model("medium")


class DeviceFallbackTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"model": "medium", "task": "transcribe"}
        for target, value in (
            ("faster_whisper.vad.get_speech_timestamps", [{"start": 0, "end": 16000}]),
            ("ctranslate2.get_cuda_device_count", 1),
            ("transcribe.prepare_mixed_model", "cached-model"),
        ):
            mock = patch(target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    @patch.object(transcribe, "run_mixed_model")
    def test_auto_retries_same_model_on_cpu(self, run):
        run.side_effect = [RuntimeError("CUDA inference failed"), ["CPU transcript"]]
        self.assertEqual(transcribe.infer_mixed(None, self.settings, "auto"), ["CPU transcript"])
        self.assertEqual(run.call_args_list, [
            call(None, [0.0, 1.0], self.settings, "cuda", "cached-model"),
            call(None, [0.0, 1.0], self.settings, "cpu", "cached-model"),
        ])
        transcribe.prepare_mixed_model.assert_called_once()

    @patch.object(transcribe, "run_mixed_model", side_effect=RuntimeError("CUDA inference failed"))
    def test_explicit_cuda_reports_failure_without_cpu_retry(self, run):
        with self.assertRaisesRegex(SystemExit, "repair CUDA 12"):
            transcribe.infer_mixed(None, self.settings, "cuda")
        run.assert_called_once()

    @patch.object(transcribe, "run_mixed_model", return_value=["CPU transcript"])
    def test_cpu_preference_avoids_cuda(self, run):
        self.assertEqual(transcribe.infer_mixed(None, self.settings, "cpu"), ["CPU transcript"])
        self.assertEqual(run.call_args.args[3], "cpu")

    @patch.object(transcribe, "run_mixed_model")
    def test_download_failure_does_not_trigger_cuda_retry(self, run):
        transcribe.prepare_mixed_model.side_effect = SystemExit("Download failed")
        with self.assertRaisesRegex(SystemExit, "Download failed"):
            transcribe.infer_mixed(None, self.settings, "auto")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
