import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from PIL import Image
from image_analyzer.generator import generation_options_from_meta, original_output_path

from image_analyzer import AnalysisError, ImageAnalyzer, ImageGenerator, prompt_from_meta
from image_analyzer.cli import main
from test_analyzer import PNG, server


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.meta = self.root / "image.jpg.json"
        self.meta.write_text(json.dumps({"schema_version": 1, "analysis": {
            "description": 'A red square; $(echo example) "quoted"',
            "scene": "White background"}}))
        for name in ("diffusion.gguf", "encoder.gguf", "vae.safetensors"):
            (self.root / name).touch()
        self.runner = self.root / "sd-cli"
        self.runner.write_text(f"#!{sys.executable}\n"
                               "import sys, pathlib, json\n"
                               "from PIL import Image, ImageDraw\n"
                               "args = sys.argv[1:]\n"
                               "assert '$(echo example)' in args[args.index('-p') + 1]\n"
                               "pathlib.Path(__file__).with_name('args.json').write_text(json.dumps(args))\n"
                               "w, h = int(args[args.index('-W')+1]), int(args[args.index('-H')+1])\n"
                               "image = Image.new('RGBA', (w,h), (0,0,0,0))\n"
                               "ImageDraw.Draw(image).rectangle((w//4,h//4,w*3//4,h*3//4), fill='red')\n"
                               "image.save(args[args.index('-o')+1])\n"
                               "pathlib.Path(__file__).with_name('model-output.png').write_bytes("
                               "pathlib.Path(args[args.index('-o')+1]).read_bytes())\n")
        self.runner.chmod(0o755)
        self.generator = ImageGenerator(self.root / "diffusion.gguf", self.root / "encoder.gguf",
                                        self.root / "vae.safetensors", executable=str(self.runner))

    def test_generate_runs_executable_and_publishes_image(self):
        result = self.generator.generate(self.meta)
        self.assertEqual(result.name, "image.jpg.json.generated.png")
        with Image.open(result) as image:
            self.assertEqual(image.size, (1024, 1024))
            self.assertEqual(image.mode, "RGB")
        self.assertFalse(original_output_path(result).exists())
        with self.assertRaises(FileExistsError):
            self.generator.generate(self.meta)
        self.generator.generate(self.meta, overwrite=True)
        with Image.open(result) as image:
            self.assertEqual(image.size, (1024, 1024))

    def test_failure_preserves_existing_output_and_cleans_temporary_files(self):
        output = self.root / "output.png"
        output.write_bytes(PNG)
        self.runner.write_text("#!/usr/bin/env python3\nraise SystemExit(2)\n")
        with self.assertRaisesRegex(AnalysisError, "sd-cli failed"):
            self.generator.generate(self.meta, output, overwrite=True)
        self.assertEqual(output.read_bytes(), PNG)
        self.assertFalse(list(self.root.glob(".image-generation-*")))

    def test_keep_original_preserves_raw_bytes_and_render_dimensions(self):
        metadata = json.loads(self.meta.read_text())
        metadata["image"] = {"width": 164, "height": 142, "transparency": False}
        self.meta.write_text(json.dumps(metadata))
        output = self.generator.generate(self.meta, keep_original=True)
        original = original_output_path(output)
        self.assertEqual(original.name, "image.jpg.json.generated.original.png")
        self.assertEqual(original.read_bytes(), (self.root / "model-output.png").read_bytes())
        with Image.open(original) as image:
            self.assertEqual(image.size, (192, 160))
            self.assertEqual(image.mode, "RGBA")
        with Image.open(output) as image:
            self.assertEqual(image.size, (164, 142))
            self.assertEqual(image.mode, "RGB")

    def test_existing_original_prevents_inference_without_overwrite(self):
        output = self.root / "custom.png"
        original = original_output_path(output)
        original.write_bytes(b"keep this")
        with self.assertRaises(FileExistsError):
            self.generator.generate(self.meta, output, keep_original=True)
        self.assertEqual(original.read_bytes(), b"keep this")
        self.assertFalse((self.root / "args.json").exists())
        self.generator.generate(self.meta, output, keep_original=True, overwrite=True)
        self.assertEqual(original.read_bytes(), (self.root / "model-output.png").read_bytes())

    def test_independent_render_and_final_resolution(self):
        metadata = json.loads(self.meta.read_text())
        metadata["image"] = {"width": 164, "height": 142, "transparency": True}
        metadata["generation"] = {"width": 1024, "height": 896}
        self.meta.write_text(json.dumps(metadata))
        output = self.generator.generate(self.meta, keep_original=True)
        args = json.loads((self.root / "args.json").read_text())
        self.assertEqual(args[args.index("-W")+1], "1024")
        self.assertEqual(args[args.index("-H")+1], "896")
        with Image.open(original_output_path(output)) as image:
            self.assertEqual(image.size, (1024, 896))
            self.assertEqual(image.mode, "RGBA")
        with Image.open(output) as image:
            self.assertEqual(image.size, (164, 142))
            self.assertEqual(image.mode, "RGBA")

    def test_invalid_render_resolution_prevents_inference(self):
        metadata = json.loads(self.meta.read_text())
        for generation in (None, [], {"width": 1024}, {"width": True, "height": 1024},
                           {"width": 1000, "height": 1024}, {"width": -32, "height": 1024},
                           {"width": 1024.0, "height": 1024}):
            with self.subTest(generation=generation):
                self.meta.write_text(json.dumps({**metadata, "generation": generation}))
                with self.assertRaises(ValueError):
                    self.generator.generate(self.meta)
        self.assertFalse((self.root / "args.json").exists())

    def test_bad_metadata_and_dimensions(self):
        for value in ({}, [], {"schema_version": 1, "analysis": {"description": ""}}):
            self.meta.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                prompt_from_meta(self.meta)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.generator.generate(self.meta, width=0)

    def test_metadata_resolution_and_native_transparency(self):
        metadata = json.loads(self.meta.read_text())
        metadata["image"] = {"width": 164, "height": 142, "transparency": True}
        self.meta.write_text(json.dumps(metadata))
        output = self.generator.generate(self.meta, width=1024, height=1024)
        args = json.loads((self.root / "args.json").read_text())
        self.assertEqual(args[args.index("-W")+1], "192")
        self.assertEqual(args[args.index("-H")+1], "160")
        self.assertIn("RGBA image with transparency", args[args.index("-p")+1])
        with Image.open(output) as image:
            self.assertEqual(image.size, (164, 142))
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getchannel("A").getextrema(), (0, 255))

    def test_invalid_dimensions_or_transparency_rejected_before_runner(self):
        original = json.loads(self.meta.read_text())
        for image in ({"width": 164}, {"width": True, "height": 32},
                      {"width": -1, "height": 32}, {"width": 1.5, "height": 32},
                      {"transparency": "true"}):
            with self.subTest(image=image):
                self.meta.write_text(json.dumps({**original, "image": image}))
                with self.assertRaises(ValueError):
                    self.generator.generate(self.meta)
        self.assertFalse((self.root / "args.json").exists())

    def test_missing_fields_use_legacy_defaults(self):
        self.assertEqual(generation_options_from_meta(self.meta, width=512, height=256), (512, 256, False))

    def test_opaque_model_result_fails_transparency_request(self):
        metadata = json.loads(self.meta.read_text())
        metadata["image"] = {"width": 32, "height": 32, "transparency": True}
        self.meta.write_text(json.dumps(metadata))
        self.runner.write_text(f"#!{sys.executable}\nimport sys\nfrom PIL import Image\n"
                               "Image.new('RGB', (32,32), 'white').save(sys.argv[sys.argv.index('-o')+1])\n")
        with self.assertRaisesRegex(AnalysisError, "transparent image"):
            self.generator.generate(self.meta)
        self.assertFalse(Path(f"{self.meta}.generated.png").exists())

    def test_prompt_extraction_does_not_contact_lm_studio(self):
        with patch.object(ImageAnalyzer, "_request", side_effect=AssertionError("unexpected network")):
            self.assertIn("A red square", prompt_from_meta(self.meta))

    def test_cli_generation(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"generation": {
            "diffusion_model": "diffusion.gguf", "text_encoder": "encoder.gguf",
            "vae": "vae.safetensors", "sd_cli": "./sd-cli", "unload_lm_studio": False}}))
        with patch.dict(os.environ, {"IMAGE_ANALYZER_CONFIG": str(config)}):
            self.assertEqual(main([str(self.meta), "--keep-original"]), 0)
        self.assertTrue(Path(f"{self.meta}.generated.png").exists())
        self.assertTrue(original_output_path(f"{self.meta}.generated.png").exists())

    def test_timeout_and_missing_output(self):
        self.runner.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n")
        self.generator.timeout = 0.05
        with self.assertRaisesRegex(AnalysisError, "timed out"):
            self.generator.generate(self.meta)
        self.generator.timeout = 5
        self.runner.write_text("#!/usr/bin/env python3\npass\n")
        with self.assertRaisesRegex(AnalysisError, "without creating"):
            self.generator.generate(self.meta)
        self.assertFalse(Path(f"{self.meta}.generated.png").exists())


class ModelManagementTests(unittest.TestCase):
    def test_native_routes_switch_instances_and_use_returned_identifier(self):
        inventory = {"models": [
            {"key": "vision", "type": "llm", "capabilities": {"vision": True}, "loaded_instances": []},
            {"key": "other", "loaded_instances": [{"id": "other-instance"}]},
        ]}

        def respond(path, payload):
            if path == "/api/v1/models":
                return inventory
            if path.endswith("/unload"):
                return {"instance_id": payload["instance_id"]}
            return {"status": "loaded", "instance_id": "vision-instance"}

        with server(respond) as (analyzer, requests):
            analyzer.model = "vision"
            self.assertEqual(analyzer.load_model(unload_others=True), "vision-instance")
        self.assertEqual(requests, [
            ("/api/v1/models", None),
            ("/api/v1/models/unload", {"instance_id": "other-instance"}),
            ("/api/v1/models/load", {"model": "vision"})])
        self.assertEqual(analyzer.model, "vision-instance")

    def test_reuse_instance_without_reload(self):
        inventory = {"models": [{"key": "vision", "type": "llm", "capabilities": {"vision": True},
                                 "loaded_instances": [{"id": "existing"}]}]}
        with server(inventory) as (analyzer, requests):
            analyzer.model = "vision"
            self.assertEqual(analyzer.load_model(), "existing")
        self.assertEqual(len(requests), 1)

    def test_unknown_or_nonvision_model_does_not_unload_anything(self):
        inventory = {"models": [{"key": "diffusion", "type": "llm", "capabilities": {"vision": False},
                                 "loaded_instances": [{"id": "other"}]}]}
        for name in ("diffusion", "missing"):
            with server(inventory) as (analyzer, requests):
                analyzer.model = name
                with self.assertRaises(AnalysisError):
                    analyzer.load_model(unload_others=True)
                self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
