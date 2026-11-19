"""Check audio conditioning and vocabulary/decoding contracts without downloading models."""

import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location("accuracy_transcribe", Path(__file__).resolve().parents[1] / "transcribe.py")
transcribe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcribe)


class QualitySettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "settings.json"
        whisper = ModuleType("whisper")
        whisper.available_models = lambda: ["small"]
        tokenizer = ModuleType("whisper.tokenizer")
        tokenizer.LANGUAGES = {"es": "spanish"}
        faster = ModuleType("faster_whisper")
        faster.available_models = lambda: ["small"]
        modules = patch.dict(sys.modules, {"whisper": whisper, "whisper.tokenizer": tokenizer, "faster_whisper": faster})
        modules.start()
        self.addCleanup(modules.stop)

    def load(self, **options):
        self.path.write_text(json.dumps({"model": "small", "language": "es", "task": "transcribe",
                                         "output_format": "srt", **options}), encoding="utf-8")
        return transcribe.load_settings(self.path)

    def test_older_presets_do_not_enable_new_audio_or_decoder_processing(self):
        settings = self.load()
        self.assertFalse(settings.get("normalize_audio", False))
        self.assertTrue(transcribe.decoding_options(settings)["condition_on_previous_text"])
        self.assertEqual(settings["vocabulary_terms"], [])

    def test_options_reject_wrong_types_and_unknown_keys_before_inference(self):
        for key, value in (("normalize_audio", "true"), ("condition_on_previous_text", 0),
                           ("hallucination_filter", None), ("decoding_preset", "fastest"),
                           ("vocabulary", ["Ainhoa"]), ("glossary_path", True), ("unknown", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(**{key: value})

    def test_user_names_have_priority_and_glossary_supports_bom_comments_and_accents(self):
        (self.root / "terms.txt").write_text("# ignored\nImmunotec\nAINHOA\nglutatión\n\n", encoding="utf-8-sig")
        with patch.object(transcribe, "ROOT", self.root):
            settings = self.load(vocabulary="Ainhoa; Tzeitel\nAinhoa", glossary_path="terms.txt")
        self.assertEqual(settings["vocabulary_terms"], ["Ainhoa", "Tzeitel", "Immunotec", "glutatión"])

    def test_missing_or_oversized_glossary_fails_early(self):
        with self.assertRaises(OSError):
            self.load(glossary_path=str(self.root / "missing.txt"))
        path = self.root / "large.txt"
        path.write_text("x" * 65537, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            self.load(glossary_path=str(path))


class VocabularyTests(unittest.TestCase):
    def test_whole_phrase_limit_preserves_names_and_reports_omissions(self):
        settings = {"vocabulary_terms": ["Ainhoa", "Tzeitel", "x" * 201, "Immunotec"]}
        with patch("sys.stdout", new_callable=io.StringIO) as log:
            prompt = transcribe.vocabulary_prompt(settings, list)
        self.assertEqual(prompt, "Ainhoa, Tzeitel, Immunotec")
        self.assertLessEqual(len(" " + prompt), transcribe.PROMPT_TOKEN_LIMIT)
        self.assertIn("3 of 4", log.getvalue())

    def test_empty_vocabulary_does_not_construct_a_tokenizer_prompt(self):
        def unexpected(text):
            self.fail("Empty vocabulary should not tokenize.")
        self.assertIsNone(transcribe.vocabulary_prompt({}, unexpected))

    def test_default_glossary_keeps_user_names_separate_from_brand_terms(self):
        settings = json.loads(transcribe.DEFAULT_SETTINGS.read_text(encoding="utf-8-sig"))
        terms = transcribe.vocabulary_terms(settings)
        self.assertEqual(terms[:2], ["Ainhoa", "Tzeitel"])
        for term in ("Immunotec", "Immunocal", "BDN", "Bono de Desarrollo de Negocio", "VTOC"):
            self.assertIn(term, terms)


class DecodingTests(unittest.TestCase):
    def test_fast_hotwords_and_history_share_a_bounded_context(self):
        model = SimpleNamespace(max_length=448, get_prompt=lambda tokenizer, previous, **options: previous)
        transcribe.limit_prompt_history(model)
        tokenizer = SimpleNamespace(encode=list)
        history = list(range(1000))
        hints = "x" * 199
        previous = model.get_prompt(tokenizer, history, hotwords=hints)
        self.assertEqual(previous, history[-23:])
        self.assertEqual(len(previous) + len(tokenizer.encode(" " + hints)), 223)
        self.assertEqual(model.get_prompt(tokenizer, history), history[-223:])

    def test_accuracy_uses_beams_and_resets_context_without_disabling_fallback(self):
        for language in ("es", "multilingual"):
            options = transcribe.decoding_options({"decoding_preset": "accuracy", "language": language})
            self.assertEqual(options["beam_size"], 5)
            self.assertEqual(options["best_of"], 5)
            self.assertFalse(options["condition_on_previous_text"])
            self.assertNotIn("temperature", options)
            self.assertNotIn("no_speech_threshold", options)

    def test_explicit_context_choice_and_legacy_mixed_mode_are_preserved(self):
        self.assertTrue(transcribe.decoding_options({"decoding_preset": "accuracy",
                                                    "condition_on_previous_text": True})["condition_on_previous_text"])
        self.assertFalse(transcribe.decoding_options({"language": "multilingual"})["condition_on_previous_text"])

    def test_silence_filter_computes_alignment_even_for_text_only_output(self):
        options = transcribe.decoding_options({"word_timestamps": False, "hallucination_filter": True})
        self.assertTrue(options["word_timestamps"])
        self.assertEqual(options["hallucination_silence_threshold"], 2.0)

    def test_vocabulary_and_accuracy_reach_standard_backend(self):
        try:
            import whisper
        except ModuleNotFoundError:
            self.skipTest("Standard backend is installed in venv and venv-cuda.")
        with patch.object(whisper, "load_model") as load:
            load.return_value.to.return_value.is_multilingual = True
            transcribe.run_model("audio", {"model": "small", "language": "es", "task": "transcribe",
                                          "decoding_preset": "accuracy", "vocabulary": "Ainhoa, Tzeitel",
                                          "hallucination_filter": True}, "cpu")
        options = load.return_value.to.return_value.transcribe.call_args.kwargs
        self.assertIn("Tzeitel", options["initial_prompt"])
        self.assertTrue(options["carry_initial_prompt"])
        self.assertEqual(options["beam_size"], 5)
        self.assertTrue(options["word_timestamps"])

    def test_vocabulary_and_accuracy_reach_every_mixed_backend_window(self):
        try:
            import faster_whisper
        except ModuleNotFoundError:
            self.skipTest("Faster backend is installed in venv-fast.")
        with patch.object(faster_whisper, "WhisperModel") as model:
            model.return_value.hf_tokenizer.encode.side_effect = lambda text: SimpleNamespace(ids=list(text))
            model.return_value.transcribe.return_value = ([], SimpleNamespace(duration=0))
            transcribe.run_mixed_model("audio", [0, 1], {"model": "small", "task": "transcribe",
                                       "decoding_preset": "accuracy", "vocabulary": "Ainhoa, Tzeitel"},
                                       "cpu", "cached-model")
        options = model.return_value.transcribe.call_args.kwargs
        self.assertEqual(options["hotwords"], "Ainhoa, Tzeitel")
        self.assertNotIn("initial_prompt", options)  # Hotwords persist after context resets.
        self.assertEqual(options["clip_timestamps"], [0, 1])
        self.assertEqual(options["beam_size"], 5)


class NormalizationTests(unittest.TestCase):
    def voice(self, seconds=2, amplitude=0.02):
        time = np.arange(round(seconds * transcribe.SAMPLE_RATE)) / transcribe.SAMPLE_RATE
        return (amplitude * (np.sin(2 * np.pi * 220 * time) + 0.2 * np.sin(2 * np.pi * 440 * time))).astype(np.float32)

    def test_disabled_normalization_is_a_true_bypass(self):
        audio = self.voice()
        with patch.object(transcribe, "normalize_region", side_effect=AssertionError("Must not process")):
            self.assertIs(transcribe.prepare_audio(audio, {}), audio)

    def test_quiet_clean_speech_is_raised_without_noise_reduction_or_timing_changes(self):
        audio = self.voice()
        result, gain, reduced = transcribe.normalize_region(audio)
        self.assertFalse(reduced)
        self.assertGreater(gain, 3.9)
        self.assertLessEqual(gain, 10 ** (12 / 20))
        np.testing.assert_allclose(result, audio * gain)
        normalized = transcribe.prepare_audio(audio, {"normalize_audio": True})
        self.assertEqual(normalized.shape, audio.shape)
        self.assertEqual(normalized.dtype, audio.dtype)
        np.testing.assert_allclose(normalized, result, atol=1e-7)

    def test_loud_audio_silence_noise_only_and_short_clips_are_left_alone(self):
        rng = np.random.default_rng(42)
        for audio in (self.voice(amplitude=0.3), np.zeros(16000, dtype=np.float32),
                      rng.normal(0, 0.01, 32000).astype(np.float32), self.voice(seconds=0.01)):
            with self.subTest(samples=audio.size):
                result, gain, reduced = transcribe.normalize_region(audio)
                np.testing.assert_array_equal(result, audio)
                self.assertEqual(gain, 1)
                self.assertFalse(reduced)

    def test_stationary_noise_is_reduced_only_when_boosting(self):
        rng = np.random.default_rng(42)
        voice = self.voice()
        voice[:8000] = 0
        noise = rng.normal(0, 0.0012, voice.size).astype(np.float32)
        audio = voice + noise
        result, gain, reduced = transcribe.normalize_region(audio)
        self.assertTrue(reduced)
        self.assertGreater(gain, 1)
        self.assertLess(np.std(result[1000:7000]), np.std(audio[1000:7000]) * gain * 0.85)
        # The same signal at a normal level must not be denoised.
        loud = audio * 15
        with patch.object(np.fft, "irfft", side_effect=AssertionError("Must not denoise")):
            result, gain, reduced = transcribe.normalize_region(loud)
        np.testing.assert_array_equal(result, loud)
        self.assertEqual(gain, 1)
        self.assertFalse(reduced)

    def test_overlap_boundaries_and_peaks_stay_bounded(self):
        audio = self.voice(seconds=17.013)
        audio[5 * transcribe.SAMPLE_RATE] = 0.9
        result = transcribe.prepare_audio(audio, {"normalize_audio": True})
        self.assertEqual(result.size, audio.size)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertLessEqual(float(np.max(np.abs(result))), 0.950001)
        for boundary in (4, 8, 12, 16):
            index = boundary * transcribe.SAMPLE_RATE
            self.assertLess(float(np.max(np.abs(np.diff(result[index - 10:index + 10])))), 0.02)

    def test_invalid_waveform_is_rejected(self):
        for audio in (np.array([np.nan], dtype=np.float32), np.ones((2, 512), dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "finite mono"):
                transcribe.prepare_audio(audio, {"normalize_audio": True})


if __name__ == "__main__":
    unittest.main()
