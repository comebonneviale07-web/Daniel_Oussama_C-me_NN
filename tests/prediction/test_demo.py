"""Offline image validation and Streamlit interaction checks, with no training or downloads."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from src import demo


def app():
    from src.demo import render
    render()


class DemoTests(unittest.TestCase):
    def tearDown(self):
        demo.cached_model.clear()

    def test_image_decoding_handles_orientation_and_rejects_bad_files(self):
        buffer = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6  # camera rotated clockwise
        Image.new("RGB", (20, 10), "red").save(buffer, format="JPEG", exif=exif)
        decoded = demo.read_image(buffer.getvalue())
        self.assertEqual(decoded.size, (10, 20))
        self.assertEqual(decoded.mode, "RGB")
        for content in (b"", b"not a painting", b"x" * (demo.MAX_BYTES + 1)):
            with self.assertRaises(ValueError):
                demo.read_image(content)
        with patch.object(demo, "MAX_PIXELS", 1):
            with self.assertRaisesRegex(ValueError, "resize"):
                demo.read_image(buffer.getvalue())

    def test_every_style_has_a_curated_description(self):
        descriptions = demo.load_descriptions()
        self.assertEqual(set(descriptions), {"Baroque", "Cubism", "Impressionism", "Pop_Art", "Ukiyo_e"})
        for info in descriptions.values():
            self.assertTrue(info["label"] and info["description"])
            self.assertEqual(len(info["look_for"]), 3)

    def test_prediction_reuses_existing_pipeline_and_flags_low_scores(self):
        bundle = (object(), "m3_augmented", list(demo.load_descriptions()))
        image = Image.new("RGB", (32, 32))
        with patch.object(demo.inference, "predict", return_value=[("Baroque", .4), ("Cubism", .35), ("Pop_Art", .15)]) as predict:
            result = demo.analyse_image(image, bundle)
        predict.assert_called_once_with(*bundle, image, top=3)
        self.assertTrue(result["tentative"])
        self.assertEqual(result["model"], "m3_augmented")
        self.assertEqual(len(result["predictions"]), 3)
        self.assertGreaterEqual(result["inference_ms"], 0)

    def test_app_example_shows_predictions_explanation_and_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "config.json").write_text(json.dumps({"model": "m3_augmented"}), encoding="utf-8")
            (run / "best.pt").write_bytes(b"model is mocked")
            image = run / "example.png"
            Image.new("RGB", (32, 32), "red").save(image)
            bundle = (object(), "m3_augmented", list(demo.load_descriptions()))
            with patch.dict(os.environ, {"MUSEUM_DEMO_RUN": str(run)}), \
                 patch.object(demo, "example_path", return_value=image), \
                 patch.object(demo.inference, "load_run", return_value=bundle), \
                 patch.object(demo.inference, "predict", return_value=[("Baroque", .7), ("Cubism", .2), ("Pop_Art", .05)]):
                view = AppTest.from_function(app, default_timeout=30).run()
                self.assertEqual(len(view.exception), 0)
                self.assertEqual(len(view.get("file_uploader")), 1)
                view.button[0].click().run()
                self.assertEqual(len(view.exception), 0)
                self.assertTrue(any(header.value == "Baroque" for header in view.header))
                self.assertEqual(len(view.get("progress")), 3)
                self.assertEqual(len(view.get("download_button")), 1)
                self.assertTrue(any("training collection" in caption.value for caption in view.caption))

    def test_missing_checkpoint_is_a_readable_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "config.json").write_text(json.dumps({"model": "m3_augmented"}), encoding="utf-8")
            with patch.dict(os.environ, {"MUSEUM_DEMO_RUN": str(run)}):
                view = AppTest.from_function(app, default_timeout=30).run()
                self.assertEqual(len(view.exception), 0)
                self.assertEqual(len(view.error), 1)
                self.assertIn("trained model", view.error[0].value)


if __name__ == "__main__":
    unittest.main()
