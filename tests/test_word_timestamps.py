"""Check word timing settings and outputs without downloading or running models.

Run in both venv and venv-fast to exercise each backend's installed output writers.
"""

from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("word_timing_transcribe", Path(__file__).resolve().parents[1] / "transcribe.py")
transcribe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcribe)


def sample_segments():
    return [{
        "start": 1.0, "end": 2.5, "text": " Ahoj světe!",
        "words": [
            {"start": 1.0, "end": 1.4, "word": " Ahoj", "probability": 0.9},
            {"start": 1.6, "end": 2.5, "word": " světe!", "probability": 0.8},
        ],
    }]


WORD_SRT = "1\n00:00:01,000 --> 00:00:01,400\nAhoj\n\n2\n00:00:01,600 --> 00:00:02,500\nsvěte!\n\n"


class SettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "settings.json"
        # Catalog stubs let both environments check settings for both language modes.
        whisper = ModuleType("whisper")
        whisper.available_models = lambda: ["small"]
        tokenizer = ModuleType("whisper.tokenizer")
        tokenizer.LANGUAGES = {"cs": "czech"}
        faster = ModuleType("faster_whisper")
        faster.available_models = lambda: ["small"]
        modules = patch.dict(sys.modules, {"whisper": whisper, "whisper.tokenizer": tokenizer, "faster_whisper": faster})
        modules.start()
        self.addCleanup(modules.stop)

    def load(self, output_format, language=None, **options):
        self.path.write_text(json.dumps({
            "model": "small", "language": language, "task": "transcribe",
            "output_format": output_format, **options,
        }), encoding="utf-8")
        return transcribe.load_settings(self.path)

    def test_word_timing_is_enabled_for_supported_formats_in_both_backends(self):
        for language in (None, "multilingual"):
            for output_format in ("srt", "vtt", "json", "all"):
                with self.subTest(language=language, format=output_format):
                    self.assertTrue(self.load(output_format, language, word_timestamps=True)["word_timestamps"])

    def test_text_formats_skip_alignment_without_changing_saved_preference(self):
        for output_format in ("txt", "tsv"):
            with self.subTest(format=output_format):
                self.assertFalse(self.load(output_format, word_timestamps=True)["word_timestamps"])
                self.assertTrue(json.loads(self.path.read_text(encoding="utf-8"))["word_timestamps"])

    def test_old_settings_and_disabled_option_keep_segment_timing(self):
        self.assertFalse(self.load("srt")["word_timestamps"])
        self.assertFalse(self.load("all", word_timestamps=False)["word_timestamps"])

    def test_word_timing_requires_a_json_boolean(self):
        for value in ("true", 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "true or false"):
                self.load("srt", word_timestamps=value)


class StandardOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import whisper
        except ModuleNotFoundError:
            raise unittest.SkipTest("Run in the standard Whisper venv to check its writers.")
        cls.whisper = whisper

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.media = Path(temporary.name) / "sample.wav"
        self.media.touch()
        self.result = {"segments": sample_segments(), "text": " Ahoj světe!", "language": "cs"}

    def run_output(self, output_format, enabled=True):
        settings = {"model": "small", "language": "cs", "task": "transcribe",
                    "output_format": output_format, "word_timestamps": enabled}
        with patch.object(self.whisper, "load_audio", return_value="decoded"), \
                patch.object(transcribe, "infer", return_value=self.result):
            return transcribe.transcribe(self.media, settings, "cpu")

    def test_srt_has_one_cue_per_word(self):
        output = self.run_output("srt")
        self.assertEqual((output / "sample.srt").read_text(encoding="utf-8"), WORD_SRT)

    def test_all_formats_include_word_srt_inline_vtt_and_json_timings(self):
        output = self.run_output("all")
        self.assertEqual({path.suffix for path in output.iterdir()}, {".txt", ".srt", ".vtt", ".tsv", ".json"})
        self.assertEqual((output / "sample.srt").read_text(encoding="utf-8"), WORD_SRT)
        self.assertIn("Ahoj<00:00:01.600> světe!", (output / "sample.vtt").read_text(encoding="utf-8"))
        result = json.loads((output / "sample.json").read_text(encoding="utf-8"))
        self.assertEqual(result["segments"][0]["words"], sample_segments()[0]["words"])
        self.assertEqual((output / "sample.txt").read_text(encoding="utf-8"), "Ahoj světe!\n")
        self.assertIn("1000\t2500\tAhoj světe!", (output / "sample.tsv").read_text(encoding="utf-8"))

    def test_disabled_timing_keeps_one_cue_per_segment(self):
        del self.result["segments"][0]["words"]
        output = self.run_output("all", enabled=False)
        self.assertEqual((output / "sample.srt").read_text(encoding="utf-8"),
                         "1\n00:00:01,000 --> 00:00:02,500\nAhoj světe!\n\n")
        self.assertNotIn("<00:", (output / "sample.vtt").read_text(encoding="utf-8"))

    def test_alignment_option_reaches_standard_model(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(self.whisper, "load_model") as load_model:
                transcribe.run_model("decoded", {"model": "small", "language": "cs", "task": "transcribe",
                                                "word_timestamps": enabled}, "cpu")
                self.assertIs(load_model.return_value.to.return_value.transcribe.call_args.kwargs["word_timestamps"], enabled)


@dataclass
class MixedSegment:
    start: float
    end: float
    text: str
    words: list | None


class MixedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import faster_whisper
        except ModuleNotFoundError:
            raise unittest.SkipTest("Run in venv-fast to check faster-whisper outputs.")
        cls.faster = faster_whisper

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.staging = Path(temporary.name)
        self.segments = [MixedSegment(**segment) for segment in sample_segments()]

    def run_output(self, output_format, enabled=True):
        transcribe.write_mixed_outputs(self.segments, Path("sample.wav"), self.staging, output_format, enabled)

    def test_srt_has_one_cue_per_word(self):
        self.run_output("srt")
        self.assertEqual((self.staging / "sample.srt").read_text(encoding="utf-8"), WORD_SRT)

    def test_all_formats_include_word_srt_inline_vtt_and_json_timings(self):
        self.run_output("all")
        self.assertEqual({path.suffix for path in self.staging.iterdir()}, {".txt", ".srt", ".vtt", ".tsv", ".json"})
        self.assertEqual((self.staging / "sample.srt").read_text(encoding="utf-8"), WORD_SRT)
        self.assertIn("Ahoj<00:00:01.600> světe!", (self.staging / "sample.vtt").read_text(encoding="utf-8"))
        result = json.loads((self.staging / "sample.json").read_text(encoding="utf-8"))
        self.assertEqual(result["segments"][0]["words"], sample_segments()[0]["words"])
        self.assertEqual((self.staging / "sample.txt").read_text(encoding="utf-8"), "Ahoj světe!\n")
        self.assertIn("1000\t2500\tAhoj světe!", (self.staging / "sample.tsv").read_text(encoding="utf-8"))

    def test_disabled_timing_keeps_one_cue_per_segment(self):
        self.segments[0].words = None
        self.run_output("all", enabled=False)
        self.assertEqual((self.staging / "sample.srt").read_text(encoding="utf-8"),
                         "1\n00:00:01,000 --> 00:00:02,500\nAhoj světe!\n\n")
        self.assertNotIn("<00:", (self.staging / "sample.vtt").read_text(encoding="utf-8"))

    def test_alignment_option_reaches_mixed_model(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(self.faster, "WhisperModel") as model:
                model.return_value.transcribe.return_value = ([], SimpleNamespace(duration=0))
                transcribe.run_mixed_model("decoded", [0.0, 2.5],
                                           {"model": "small", "task": "transcribe", "word_timestamps": enabled},
                                           "cpu", "cached-model")
                self.assertIs(model.return_value.transcribe.call_args.kwargs["word_timestamps"], enabled)


class SubtitleEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from faster_whisper.utils import format_timestamp
        except ModuleNotFoundError:
            from whisper.utils import format_timestamp
        cls.format_timestamp = staticmethod(format_timestamp)

    def test_missing_word_data_falls_back_and_empty_text_does_not_break_numbering(self):
        output = io.StringIO()
        transcribe.write_word_srt([
            {"start": 0.0, "end": 0.5, "text": " "},
            {"start": 1.0, "end": 2.0, "text": " Fallback --> text\n", "words": []},
            {"start": 2.0, "end": 3.0, "text": "Also fallback", "words": None},
        ], output, self.format_timestamp)
        self.assertEqual(output.getvalue(), "1\n00:00:01,000 --> 00:00:02,000\nFallback -> text\n\n"
                                           "2\n00:00:02,000 --> 00:00:03,000\nAlso fallback\n\n")

    def test_zero_duration_word_gets_a_positive_cue_duration(self):
        segments = sample_segments()
        segments[0]["words"] = [{"start": 1.0001, "end": 1.0002, "word": " krátce"}]
        output = io.StringIO()
        transcribe.write_word_srt(segments, output, self.format_timestamp)
        self.assertEqual(output.getvalue(), "1\n00:00:01,000 --> 00:00:01,001\nkrátce\n\n")


if __name__ == "__main__":
    unittest.main()
