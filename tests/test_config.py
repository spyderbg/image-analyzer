import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from image_analyzer import ImageAnalyzer, process_file
from image_analyzer.cli import main
from image_analyzer.config import load_config
from test_analyzer import PNG, completion, server


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "settings.json"
        self.config.write_text("{}")
        self.environment = patch.dict(os.environ, {"IMAGE_ANALYZER_CONFIG": str(self.config)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_defaults_partial_overrides_and_relative_paths(self):
        self.config.write_text(json.dumps({"generation": {
            "diffusion_model": "models/test.gguf", "sd_cli": "./bin/sd-cli"}}))
        settings = load_config()
        self.assertEqual(settings["generation"]["diffusion_model"], str(self.root / "models/test.gguf"))
        self.assertEqual(settings["generation"]["sd_cli"], str(self.root / "bin/sd-cli"))
        self.assertTrue(settings["analysis"]["load_model"])
        settings["analysis"]["model"] = "changed"
        self.assertEqual(load_config()["analysis"]["model"], "qwen/qwen3-vl-8b")

    def test_invalid_config_reports_setting_and_file(self):
        for settings in ([], {"typo": True}, {"overwrite": "false"},
                         {"analysis": {"timeout": -1}}, {"generation": {"width": 1.5}},
                         {"generation": {"seed": True}}):
            with self.subTest(settings=settings):
                self.config.write_text(json.dumps(settings))
                with self.assertRaisesRegex(ValueError, "Invalid configuration.*settings.json"):
                    load_config()

    def test_explicit_missing_config_fails(self):
        with self.assertRaisesRegex(ValueError, "Invalid configuration"):
            load_config(self.root / "missing.json")

    def test_one_argument_image_uses_configured_model_loading(self):
        image = self.root / "TEST.PNG"
        image.write_bytes(PNG)

        def respond(path, payload):
            if path == "/api/v1/models":
                return {"models": [{"key": "vision", "type": "llm", "capabilities": {"vision": True},
                                    "loaded_instances": []}]}
            if path.endswith("/load"):
                return {"status": "loaded", "instance_id": "vision-instance"}
            return completion()

        with server(respond) as (analyzer, requests):
            self.config.write_text(json.dumps({"analysis": {"base_url": analyzer.base_url, "model": "vision"}}))
            self.assertEqual(main([str(image)]), 0)
        self.assertEqual([route for route, _ in requests],
                         ["/api/v1/models", "/api/v1/models/load", "/v1/chat/completions"])
        self.assertEqual(requests[-1][1]["model"], "vision-instance")
        self.assertTrue(Path(f"{image}.json").is_file())

    def test_missing_generation_settings_do_not_unload_models(self):
        meta = self.root / "test.JSON"
        meta.write_text(json.dumps({"schema_version": 1, "analysis": {"description": "A cat"}}))
        with patch.object(ImageAnalyzer, "_request", side_effect=AssertionError("unexpected network")):
            self.assertEqual(main([str(meta)]), 1)

    def test_force_recreates_json_without_changing_config(self):
        image = self.root / "image.png"
        image.write_bytes(PNG)
        sidecar = Path(f"{image}.json")
        for flag in ("--force", "-f"):
            with self.subTest(flag=flag), server(completion()) as (analyzer, requests):
                sidecar.write_text("old content")
                self.config.write_text(json.dumps({"overwrite": False, "analysis": {
                    "base_url": analyzer.base_url, "load_model": False}}))
                original_config = self.config.read_bytes()
                self.assertEqual(main([str(image), flag]), 0)
                self.assertEqual(json.loads(sidecar.read_text())["image"]["width"], 1)
                self.assertEqual(len(requests), 1)
                self.assertEqual(self.config.read_bytes(), original_config)

    def test_skip_and_unsupported_input_do_not_contact_server(self):
        image = self.root / "image.png"
        image.write_bytes(PNG)
        Path(f"{image}.json").write_text("existing")
        unsupported = self.root / "file.txt"
        unsupported.touch()
        with patch.object(ImageAnalyzer, "_request", side_effect=AssertionError("unexpected network")):
            self.assertEqual(main([str(image)]), 0)
            self.assertEqual(main([str(unsupported)]), 1)
            with self.assertRaises(FileNotFoundError):
                process_file(self.root / "missing.png")


if __name__ == "__main__":
    unittest.main()
