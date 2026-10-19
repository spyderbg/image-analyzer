from io import BytesIO
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from PIL import Image

from image_analyzer import AnalysisError, ImageAnalyzer
from image_analyzer.cli import main


_buffer = BytesIO()
Image.new("RGB", (1, 1), "white").save(_buffer, format="PNG")
PNG = _buffer.getvalue()
ANALYSIS = {"description": "A white square.", "scene": "Plain background",
            "objects": [], "tags": ["white"], "text": ""}


@contextmanager
def server(response, status=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(response(self.path, requests[-1][1])
                                       if callable(response) else response).encode())

        def do_GET(self):
            requests.append((self.path, None))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(response(self.path, None)
                                       if callable(response) else response).encode())

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield ImageAnalyzer(base_url=f"http://127.0.0.1:{httpd.server_port}/v1"), requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def completion(analysis=ANALYSIS, finish="stop"):
    return {"model": "qwen/qwen3-vl-8b", "choices": [{
        "finish_reason": finish, "message": {"content": json.dumps(analysis)}}]}


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / "sample.png"
        self.image.write_bytes(PNG)

    def test_http_image_payload_and_sidecar(self):
        with server(completion()) as (analyzer, requests):
            meta = analyzer.write_meta(self.image)
        result = json.loads(meta.read_text())
        self.assertEqual(meta.name, "sample.png.json")
        self.assertEqual(result["analysis"], ANALYSIS)
        self.assertEqual(result["image"]["size_bytes"], len(PNG))
        self.assertEqual(len(result["image"]["sha256"]), 64)
        route, payload = requests[0]
        self.assertEqual(route, "/v1/chat/completions")
        image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
        import base64
        self.assertEqual(base64.b64decode(image_url.split(",")[1]), PNG)
        self.assertEqual(result["image"]["width"], 1)
        self.assertEqual(result["image"]["height"], 1)
        self.assertFalse(result["image"]["transparency"])
        self.assertEqual(result["generation"], {"width": 32, "height": 32})
        self.assertEqual(payload["response_format"]["type"], "json_schema")

    def test_existing_file_is_preserved_without_inference(self):
        destination = Path(f"{self.image}.json")
        destination.write_text("original")
        with patch.object(ImageAnalyzer, "analyze") as analyze:
            with self.assertRaises(FileExistsError):
                ImageAnalyzer().write_meta(self.image)
            analyze.assert_not_called()
        self.assertEqual(destination.read_text(), "original")

    def test_overwrite(self):
        destination = Path(f"{self.image}.json")
        destination.write_text("original")
        with server(completion()) as (analyzer, _):
            analyzer.write_meta(self.image, overwrite=True)
        self.assertEqual(json.loads(destination.read_text())["analysis"], ANALYSIS)

    def test_invalid_or_truncated_response_never_writes_metadata(self):
        for response in (completion({"description": "incomplete"}),
                         completion(finish="length"), {"choices": []},
                         completion({**ANALYSIS, "objects": "square"})):
            with self.subTest(response=response), server(response) as (analyzer, _):
                with self.assertRaises(AnalysisError):
                    analyzer.write_meta(self.image)
                self.assertFalse(Path(f"{self.image}.json").exists())

    def test_http_failure(self):
        with server({"error": "model not found"}, 400) as (analyzer, _):
            with self.assertRaisesRegex(AnalysisError, "HTTP 400.*model not found"):
                analyzer.write_meta(self.image)

    def test_unsupported_input_and_source_destination(self):
        invalid = self.image.with_suffix(".txt")
        invalid.write_text("not an image")
        with self.assertRaises(ValueError):
            ImageAnalyzer().analyze(invalid)
        with self.assertRaises(ValueError):
            ImageAnalyzer().write_meta(self.image, self.image, overwrite=True)
        self.assertEqual(self.image.read_bytes(), PNG)

    def test_cli_recursive_continues_after_invalid_image(self):
        subdir = self.image.parent / "nested"
        subdir.mkdir()
        (subdir / "bad.jpg").write_text("invalid")
        with server(completion()) as (analyzer, requests):
            config = self.image.parent / "config.json"
            config.write_text(json.dumps({"recursive": True, "analysis": {
                "base_url": analyzer.base_url, "load_model": False}}))
            with patch.dict(os.environ, {"IMAGE_ANALYZER_CONFIG": str(config)}):
                code = main([str(self.image.parent)])
        self.assertEqual(code, 1)
        self.assertEqual(len(requests), 1)
        self.assertTrue(Path(f"{self.image}.json").exists())

    def test_pixel_properties_for_supported_formats(self):
        for fmt, transparent in (("PNG", True), ("PNG", False), ("JPEG", False), ("WEBP", True)):
            with self.subTest(fmt=fmt, transparent=transparent):
                image = Image.new("RGBA" if transparent else "RGB", (164, 142),
                                  (255, 0, 0, 0) if transparent else "red")
                image.save(self.image, format=fmt)
                with patch.object(ImageAnalyzer, "_request", return_value=completion()):
                    result = ImageAnalyzer().analyze(self.image)
                self.assertEqual((result["image"]["width"], result["image"]["height"]), (164, 142))
                self.assertEqual(result["image"]["transparency"], transparent)

    def test_palette_transparency_and_opaque_alpha(self):
        image = Image.new("P", (23, 17), 0)
        image.save(self.image, format="PNG", transparency=0)
        with patch.object(ImageAnalyzer, "_request", return_value=completion()):
            self.assertTrue(ImageAnalyzer().analyze(self.image)["image"]["transparency"])
        Image.new("RGBA", (23, 17), (0, 0, 0, 255)).save(self.image)
        with patch.object(ImageAnalyzer, "_request", return_value=completion()):
            self.assertFalse(ImageAnalyzer().analyze(self.image)["image"]["transparency"])


if __name__ == "__main__":
    unittest.main()
