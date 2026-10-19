# Local image analyzer

One command, one input path. The file extension chooses the action:

```sh
source .venv/bin/activate

image-analyzer photo.jpg       # Creates photo.jpg.json
image-analyzer photo.jpg.json  # Creates photo.jpg.json.generated.png
image-analyzer photo.jpg --force  # Recreates photo.jpg.json
```

PNG, JPEG (`.jpg`/`.jpeg`), and WebP inputs are analyzed with your local LM Studio
vision model. `.json` inputs generate a new PNG with a local `stable-diffusion.cpp`
runner. Extensions are case insensitive. No action or model flags are required.
`python -m image_analyzer FILE` works the same way.

Use `--force` (or `-f`) to replace existing output for this command only, without
editing the configuration. It also works for directory runs and regeneration
from JSON. Failed inference preserves existing output.

Keep the model's original PNG before resizing or transparency processing:

```sh
image-analyzer photo.jpg.json --keep-original
# Existing outputs: add --force to regenerate both files.
```

This saves `photo.jpg.json.generated.original.png` at the model's render size,
alongside `photo.jpg.json.generated.png` at the exact metadata size. The original
is copied byte-for-byte, preserving the model's PNG metadata and alpha. It is
also saved when no resize is needed. Both files follow the overwrite setting.
To enable this by default, set `generation.keep_original` to true in the config.

Requires Python 3.10+. Install once from this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

You can also run the installed executable directly without activating:

```sh
/media/plamen/data/Python/image-analyzer/.venv/bin/image-analyzer photo.jpg
```

## Configure once

Edit **[image-analyzer.json](image-analyzer.json)**. On this machine, the diffusion,
text encoder, matching VAE, and `sd-cli` paths are configured. The VAE is stored in
`/media/plamen/data1/image-analyzer/models/`, and the Vulkan runner is installed in
`.tools/stable-diffusion/` (upstream release `master-975-95b3090`).
On another machine, set `generation.vae` to the matching
`qwen_image_2.1_vae_bf16.safetensors` file and `generation.sd_cli` to the installed
runner executable, as well as updating the diffusion and text encoder paths.

A minimal configuration on another machine could be:

```json
{
  "analysis": {
    "model": "qwen/qwen3-vl-8b"
  },
  "generation": {
    "diffusion_model": "/models/qwen-image-2.1-Q4_K_S.gguf",
    "text_encoder": "/models/Qwen3-VL-8B-Instruct-Q8_0.gguf",
    "vae": "/models/qwen_image_2.1_vae_bf16.safetensors",
    "sd_cli": "/tools/stable-diffusion.cpp/build/bin/sd-cli"
  }
}
```

Omitted settings use defaults. Relative model and executable paths are resolved
against the configuration file's directory, and `~` expands to your home directory.
The first applicable configuration is used, in this order:

1. File specified by the `IMAGE_ANALYZER_CONFIG` environment variable.
2. `image-analyzer.json` in the current working directory.
3. `~/.config/image-analyzer/config.json`.
4. This checkout's `image-analyzer.json` when using the editable installation.

If none exists, analysis uses defaults; generation reports the missing settings.
An explicitly selected missing or malformed config is an error. Unknown settings
and invalid value types are rejected. API authentication, if enabled in LM Studio,
uses the `LM_STUDIO_API_KEY` environment variable.

| Setting | Default | Purpose |
| --- | --- | --- |
| `overwrite` | `false` | Replace existing output when true; otherwise skip it |
| `recursive` | `false` | Include subfolders when the input is a directory |
| `analysis.base_url` | `http://localhost:1234/v1` | LM Studio API address |
| `analysis.model` | `qwen/qwen3-vl-8b` | Downloaded vision model key |
| `analysis.timeout` | `180` | Request timeout in seconds |
| `analysis.max_tokens` | `1200` | Maximum response length |
| `analysis.load_model` | `true` | Load or reuse the selected model before analysis |
| `analysis.unload_others` | `true` | Unload other instances when loading the analysis model |
| `generation.diffusion_model` | empty | Diffusion weights file |
| `generation.text_encoder` | empty | Compatible text encoder weights file |
| `generation.vae` | empty | Matching VAE weights file |
| `generation.sd_cli` | `sd-cli` | Runner executable name or path |
| `generation.timeout` | `1800` | Generation timeout in seconds |
| `generation.width`, `generation.height` | `1024` | Fallback dimensions for older metadata without width/height |
| `generation.steps` | `40` | Sampling steps |
| `generation.seed` | `42` | Random seed |
| `generation.cfg_scale` | `6.0` | Prompt guidance scale |
| `generation.unload_lm_studio` | `true` | Release LM Studio model memory before generation |
| `generation.keep_original` | `false` | Keep the unmodified model PNG alongside the final image |

## Model loading and generation setup

For analysis, run LM Studio's API server (`lms server start` or the Developer tab).
The configured model must already be downloaded and support image input. The tool
uses LM Studio's native model APIs to load it if needed and reuse loaded instances.
Set `analysis.unload_others` to false to keep other models loaded, or
`analysis.load_model` to false to leave loading to LM Studio's existing settings.

For generation, install a recent **stable-diffusion.cpp** build with Qwen-Image-2.1
support and configure all three weights files plus the executable. LM Studio's
chat API does not serve diffusion image generation. The downloaded diffusion GGUF
alone is insufficient. Follow the upstream [build instructions](https://github.com/leejet/stable-diffusion.cpp#build)
and [Qwen-Image-2.1 weights guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md).
The 2.1 VAE is required; earlier Qwen-Image VAEs are not interchangeable.

Before generation, the tool validates metadata, settings, weights, and executable,
then optionally unloads LM Studio instances to free GPU memory. The runner loads
generation weights per invocation and releases them on exit. The next image input
loads the analysis model again. Unloading affects other clients of that server;
concurrent jobs are not coordinated. Set `generation.unload_lm_studio` to false
if generation should operate without contacting LM Studio.

The default generation settings use Euler sampling and CPU offload, targeting
standard Qwen-Image-2.1 weights. Turbo variants need a different sampling schedule
and are not supported by these defaults. The tool performs no automatic downloads.

## Directory input

```sh
image-analyzer /path/to/photos
```

Directories analyze supported images only; they do not generate from `.json`
sidecars. Set `recursive` to true in the config to include subfolders. Files ending
in `.json.generated.png` are excluded from directory runs. Processing continues
after individual failures. The exit code is 1 if any input fails, otherwise 0
(including skipped existing output). The old CLI flags have been replaced by
configuration settings.

## Python API

```python
from image_analyzer import process_file

process_file("photo.jpg")       # Automatically writes metadata
process_file("photo.jpg.json")  # Automatically generates an image
```

The lower-level APIs remain available for explicit control:

```python
from image_analyzer import ImageAnalyzer, ImageGenerator, prompt_from_meta

analyzer = ImageAnalyzer(model="qwen/qwen3-vl-8b")
analyzer.load_model(unload_others=True)
metadata = analyzer.analyze("photo.jpg")  # Return a dictionary without writing
analyzer.write_meta("photo.jpg")
print(prompt_from_meta("photo.jpg.json"))

generator = ImageGenerator("diffusion.gguf", "encoder.gguf", "vae.safetensors")
generator.generate("photo.jpg.json", "recreated.png")
```

`process_file` loads configuration automatically. Existing output raises
`FileExistsError` unless `overwrite` is enabled; the CLI catches this and skips it.
Other file errors raise `OSError`, invalid settings/input raise `ValueError`, and
server or generation failures raise `image_analyzer.AnalysisError`.

## Metadata and output

`photo.jpg.json` contains UTF-8 JSON with `schema_version`, `image` (filename,
MIME type, byte count, SHA-256, width, height, transparency), `generation` (render
width and height), `model`, `created_at`, and `analysis`.
Pixel properties are read directly from the source image, not guessed by the LLM:

```json
"image": {
  "filename": "photo.png",
  "width": 164,
  "height": 142,
  "transparency": true
}
```

This shows the editable size/transparency portion; the actual object also includes
MIME type, byte count, and SHA-256. `transparency` is true when the source contains
any partially or fully transparent pixels (including palette transparency).
An alpha channel containing only opaque pixels is recorded as false.

Edit `image.width` and `image.height` to change the generated PNG's exact pixel
dimensions. They must both be positive integers. Metadata dimensions take priority
over the configuration, including when using `ImageGenerator.generate` directly.
The model's render resolution is controlled separately by the JSON's top-level
`generation` object:

```json
{
  "image": {"width": 164, "height": 142, "transparency": true},
  "generation": {"width": 1024, "height": 896}
}
```

This excerpt renders a 1024×896 image and resizes it to exactly 164×142 using
Lanczos resampling. Both generation dimensions must be positive integers divisible
by 32. Using similar render and final aspect ratios avoids stretching.
`--keep-original` retains the model's 1024×896 PNG alongside the 164×142 final PNG.

New JSON files include `generation.width` and `generation.height`, initially set
to the source dimensions rounded up to multiples of 32. Edit these to render at
a larger resolution before downscaling; very small render sizes may reduce quality.
Older JSON files without `generation` keep the previous behavior: the final image
size is rounded up to the nearest multiples of 32 for rendering. The JSON's
`generation` settings describe that image only; `image-analyzer.json` still holds
the runner settings and fallback final dimensions for legacy files.

Set `image.transparency` to true to request native transparent RGBA generation,
using the model's [recommended transparency prompt](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md#alpha-channel).
This adds an instruction to omit any solid background mentioned in the description.
Alpha is preserved when resizing and saving PNG. If the model returns fully opaque
or fully transparent/blank output, generation reports an error and leaves existing
output intact; it does not fabricate a background mask. Set it to false for opaque
RGB output (any model-generated transparency is composited on white).

Older `.json` files still work: missing dimensions use the configured fallback,
and missing transparency defaults to false. Existing sidecars are not rewritten
automatically; regenerate them with `overwrite` enabled or add the fields manually.
If a generated PNG already exists, enable `overwrite` to apply edited metadata.

The `analysis` object contains:

```json
{
  "description": "A brown dog sitting on grass beside a red ball.",
  "scene": "An outdoor grassy area in daylight.",
  "objects": ["dog", "grass", "ball"],
  "tags": ["dog", "outdoors", "grass", "red ball"],
  "text": ""
}
```

Generation combines `analysis.description` and `analysis.scene`. The result is a
new interpretation, not an exact reconstruction. Descriptions and OCR may contain
model errors. Images are sent at original size without resizing, so very large
inputs may require more model context or a smaller source image. Truncated analysis
requires a larger `analysis.max_tokens` and sufficient loaded model context.

Default outputs preserve the original image and metadata. Failed inference does
not publish partial files or replace existing output. Atomic publication without
overwrite requires filesystem hard-link support. Pillow reads pixel properties and
resizes generated output while preserving alpha. Model servers/runners are installed
separately.

## Verification

```sh
python -m unittest discover -s tests -v
```

Tests cover local HTTP requests, model switching, config validation, automatic
action selection, output protection, and generation through a test executable.
They do not require a model or GPU.
