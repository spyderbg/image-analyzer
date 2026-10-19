"""One input argument: images become metadata; metadata becomes images."""

import argparse
import os
from pathlib import Path
import sys

from .analyzer import AnalysisError, ImageAnalyzer
from .config import load_config
from .generator import (ImageGenerator, generation_options_from_meta, original_output_path,
                        prompt_from_meta, render_dimensions_from_meta)

_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def process_file(path: str | Path, settings: dict | None = None) -> Path:
    """Auto-select the action, using configuration when settings are omitted."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in _EXTENSIONS | {".json"}:
        raise ValueError("Use a PNG, JPEG, WebP image or a .json file")
    config = load_config() if settings is None else settings
    generating = suffix == ".json"
    destination = Path(f"{source}.generated.png" if generating else f"{source}.json")
    if destination.exists() and not config["overwrite"]:
        raise FileExistsError(destination)
    if generating and config["generation"]["keep_original"] and not config["overwrite"]:
        original = original_output_path(destination)
        if original.exists():
            raise FileExistsError(original)
    analysis = config["analysis"]
    analyzer = ImageAnalyzer(analysis["base_url"], analysis["model"],
                             timeout=analysis["timeout"], max_tokens=analysis["max_tokens"],
                             api_key=os.getenv("LM_STUDIO_API_KEY"))
    if not generating:
        # Reject invalid image bytes before changing loaded model state.
        with source.open("rb") as handle:
            header = handle.read(12)
        if not (header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith(b"\xff\xd8\xff")
                or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")):
            raise ValueError(f"Unsupported or invalid image: {source}")
        if analysis["load_model"]:
            analyzer.load_model(unload_others=analysis["unload_others"])
        return analyzer.write_meta(source, overwrite=config["overwrite"])
    prompt_from_meta(source)
    generation = config["generation"]
    missing = [key for key in ("diffusion_model", "text_encoder", "vae") if not generation[key]]
    if missing:
        raise ValueError("Set generation." + ", generation.".join(missing) +
                         " in image-analyzer.json before generating images")
    generator = ImageGenerator(generation["diffusion_model"], generation["text_encoder"],
                               generation["vae"], executable=generation["sd_cli"],
                               timeout=generation["timeout"])
    generator.validate()
    width, height, _ = generation_options_from_meta(source, width=generation["width"], height=generation["height"])
    generator.validate_options(width, height,
                               generation["steps"], generation["cfg_scale"])
    render_dimensions_from_meta(source, width=width, height=height)
    if generation["unload_lm_studio"]:
        analyzer.unload_all_models()
    return generator.generate(source, width=generation["width"], height=generation["height"],
                              steps=generation["steps"], seed=generation["seed"],
                              cfg_scale=generation["cfg_scale"], overwrite=config["overwrite"],
                              keep_original=generation["keep_original"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Images create .json files; .json files generate images.",
        epilog="Settings are read from image-analyzer.json. No model flags are needed.")
    parser.add_argument("path", type=Path, help="Image, .json file, or directory of images")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Recreate existing output, overriding the overwrite setting")
    parser.add_argument("--keep-original", action="store_true",
                        help="Also save the model output before resizing (JSON input only)")
    args = parser.parse_args(argv)
    if args.keep_original and args.path.suffix.lower() != ".json":
        parser.error("--keep-original requires a .json input file")
    try:
        settings = load_config()
        if args.force:
            settings["overwrite"] = True
        if args.keep_original:
            settings["generation"]["keep_original"] = True
        source = args.path.expanduser()
        if source.is_dir():
            # Directory runs analyze images only, avoiding generation from sidecars.
            files = sorted(path for path in (source.rglob("*") if settings["recursive"] else source.iterdir())
                           if path.is_file() and path.suffix.lower() in _EXTENSIONS
                           and not path.name.lower().endswith((".json.generated.png", ".meta.generated.png",
                                                               ".json.generated.original.png", ".meta.generated.original.png")))
            if not files:
                raise ValueError(f"No supported images in {source}")
        else:
            files = [source]
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    failed = False
    for source in files:
        try:
            if source.suffix.lower() == ".json" and source.is_file() and (
                settings["overwrite"] or not Path(f"{source}.generated.png").exists()
            ):
                width, height, _ = generation_options_from_meta(
                    source, width=settings["generation"]["width"], height=settings["generation"]["height"])
                render_width, render_height = render_dimensions_from_meta(source, width=width, height=height)
                print(f"Generating image from {source} at {render_width}x{render_height} "
                      f"(final size: {width}x{height}); this may take a few minutes...", file=sys.stderr)
            output = process_file(source, settings)
            print(output)
            if source.suffix.lower() == ".json" and settings["generation"]["keep_original"]:
                print(original_output_path(output))
        except FileExistsError as exc:
            print(f"Skipped (output exists): {exc}", file=sys.stderr)
        except (AnalysisError, ValueError, OSError) as exc:
            print(f"Error processing {source}: {exc}", file=sys.stderr)
            failed = True
    return int(failed)
