"""Generate images from metadata using a local stable-diffusion.cpp runner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image

from .analyzer import AnalysisError


def original_output_path(output_path: str | Path) -> Path:
    """Path for the byte-for-byte output from the model runner."""
    output = Path(output_path)
    return output.with_name(f"{output.stem}.original.png")


def _read_meta(meta_path: str | Path) -> dict:
    metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Expected a schema_version 1 .json file")
    return metadata


def _prompt(metadata: dict) -> str:
    analysis = metadata.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("Metadata must contain an analysis object")
    description = analysis.get("description")
    scene = analysis.get("scene", "")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Metadata must contain a nonempty analysis.description")
    if not isinstance(scene, str):
        raise ValueError("analysis.scene must be a string")
    return "\n".join(part.strip() for part in (description, scene) if part.strip())


def generation_options_from_meta(meta_path: str | Path, *, width: int = 1024,
                                 height: int = 1024) -> tuple[int, int, bool]:
    """Metadata dimensions take precedence; defaults support legacy sidecars."""
    image = _read_meta(meta_path).get("image", {})
    if not isinstance(image, dict):
        raise ValueError("Metadata image must be an object")
    if ("width" in image) != ("height" in image):
        raise ValueError("Metadata image must specify both width and height")
    width, height = image.get("width", width), image.get("height", height)
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("Metadata width and height must be positive integers")
    transparency = image.get("transparency", False)
    if not isinstance(transparency, bool):
        raise ValueError("Metadata image.transparency must be true or false")
    return width, height, transparency


def prompt_from_meta(meta_path: str | Path) -> str:
    """Build a prompt, including native RGBA instructions when requested."""
    metadata = _read_meta(meta_path)
    prompt = _prompt(metadata)
    _, _, transparency = generation_options_from_meta(meta_path)
    if transparency:
        prompt = ("This is an RGBA image with transparency. " + prompt +
                  " The image has alpha channel and the background is transparent. "
                  "Render the described subject only; omit any described solid background.")
    return prompt


def render_dimensions_from_meta(meta_path: str | Path, *, width: int,
                                height: int) -> tuple[int, int]:
    """Read the independent model render size; align final dimensions for legacy JSON."""
    generation = _read_meta(meta_path).get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("Metadata generation must be an object")
    if ("width" in generation) != ("height" in generation):
        raise ValueError("Metadata generation must specify both width and height")
    render_width = generation.get("width", (width + 31) // 32 * 32)
    render_height = generation.get("height", (height + 31) // 32 * 32)
    if any(type(value) is not int or value <= 0 or value % 32
           for value in (render_width, render_height)):
        raise ValueError("Metadata generation.width and generation.height must be positive integer multiples of 32")
    return render_width, render_height


class ImageGenerator:
    """Use sd-cli with explicit diffusion, text encoder, and VAE file paths.

    The runner loads the selected weights per invocation and releases them on
    exit. LM Studio is not used as a diffusion server. No downloads are performed.
    """

    def __init__(self, diffusion_model: str | Path, text_encoder: str | Path,
                 vae: str | Path, *, executable: str = "sd-cli", timeout: float = 1800):
        self.diffusion_model = Path(diffusion_model).expanduser()
        self.text_encoder = Path(text_encoder).expanduser()
        self.vae = Path(vae).expanduser()
        self.executable = executable
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Generation timeout must be positive")
        self.timeout = timeout

    def validate(self) -> None:
        """Check prerequisites before changing model memory or starting work."""
        for name, path in (("diffusion model", self.diffusion_model),
                           ("text encoder", self.text_encoder), ("VAE", self.vae)):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} file: {path}")
        if shutil.which(self.executable) is None:
            raise ValueError("sd-cli not found. Install a recent stable-diffusion.cpp "
                             "build with Qwen-Image-2.1 support and configure generation.sd_cli.")

    @staticmethod
    def validate_options(width: int, height: int, steps: int, cfg_scale: float) -> None:
        if any(type(value) is not int or value <= 0 for value in (width, height)):
            raise ValueError("width and height must be positive integers")
        if steps <= 0 or not math.isfinite(cfg_scale) or cfg_scale <= 0:
            raise ValueError("steps and cfg_scale must be positive")

    def generate(self, meta_path: str | Path, output_path: str | Path | None = None, *,
                 width: int = 1024, height: int = 1024, steps: int = 40,
                 seed: int = 42, cfg_scale: float = 6.0, overwrite: bool = False,
                 keep_original: bool = False) -> Path:
        """Generate a PNG using metadata dimensions and transparency.

        width/height are fallbacks for legacy metadata without pixel dimensions.
        generation.width/height select the render size; image.width/height select
        the final output size. Legacy JSON renders on the nearest larger 32-pixel grid.
        keep_original saves the runner's unmodified PNG beside the final output.
        """
        source = Path(meta_path)
        destination = Path(output_path) if output_path else Path(f"{source}.generated.png")
        if destination.suffix.lower() != ".png":
            raise ValueError("Generation output must use the .png extension")
        if source.resolve() == destination.resolve():
            raise ValueError("Output must differ from the metadata source")
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        original = original_output_path(destination)
        if keep_original and original.exists() and not overwrite:
            raise FileExistsError(original)
        width, height, transparency = generation_options_from_meta(source, width=width, height=height)
        self.validate_options(width, height, steps, cfg_scale)
        render_width, render_height = render_dimensions_from_meta(source, width=width, height=height)
        prompt = prompt_from_meta(source)
        self.validate()
        # Keep the final destination intact if inference fails, times out, or
        # another process creates it while sd-cli runs.
        with tempfile.TemporaryDirectory(prefix=".image-generation-", dir=destination.parent) as folder:
            temporary = Path(folder) / "generated.png"
            processed = Path(folder) / "processed.png"
            command = [self.executable, "--diffusion-model", str(self.diffusion_model.resolve()),
                       "--llm", str(self.text_encoder.resolve()), "--vae", str(self.vae.resolve()),
                       "-p", prompt, "-o", str(temporary), "-W", str(render_width), "-H", str(render_height),
                       "--steps", str(steps), "--seed", str(seed), "--cfg-scale", str(cfg_scale),
                       "--sampling-method", "euler", "--offload-to-cpu"]
            try:
                with tempfile.TemporaryFile(mode="w+b") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=self.timeout, check=False)
                    if result.returncode:
                        log.seek(0, 2)
                        log.seek(max(0, log.tell() - 3000))
                        detail = log.read().decode("utf-8", errors="replace")
                        raise AnalysisError(f"sd-cli failed ({result.returncode}): {detail}")
            except subprocess.TimeoutExpired as exc:
                raise AnalysisError("Image generation timed out; increase generation.timeout in the configuration") from exc
            if not temporary.is_file():
                raise AnalysisError("sd-cli completed without creating an image")
            with temporary.open("rb") as handle:
                if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise AnalysisError("sd-cli output is not a PNG image")
            try:
                with Image.open(temporary) as rendered:
                    rendered.load()
                    if rendered.size != (render_width, render_height):
                        raise AnalysisError("sd-cli returned unexpected image dimensions")
                    pixels = rendered.convert("RGBA")
                if transparency:
                    minimum, maximum = pixels.getchannel("A").getextrema()
                    if minimum == 255 or maximum == 0:
                        raise AnalysisError("Model did not generate a visible transparent image; retry generation")
                else:
                    background = Image.new("RGBA", pixels.size, "white")
                    pixels = Image.alpha_composite(background, pixels).convert("RGB")
                if pixels.size != (width, height):
                    pixels = pixels.resize((width, height), Image.Resampling.LANCZOS)
                pixels.save(processed, format="PNG")
            except (OSError, ValueError) as exc:
                raise AnalysisError(f"Cannot read generated image: {exc}") from exc
            if keep_original:
                if overwrite:
                    os.replace(temporary, original)
                else:
                    os.link(temporary, original)
            try:
                if overwrite:
                    os.replace(processed, destination)
                else:
                    os.link(processed, destination)
            except OSError:
                if keep_original and not overwrite and original.exists() and original.samefile(temporary):
                    original.unlink()
                raise
        return destination
