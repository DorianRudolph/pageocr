# pageocr

`pageocr` extracts Markdown from images and PDFs with [LightOnOCR-2](https://lighton.ai/lighton-blogs/lighton-opens-a-new-field-for-ai-with-lightonocr-2-document-intelligence).

Features:
- Fully local
- Excellent math support (example 1 below)
- Extract figures from bounding boxes (example 2 below)
- Cross-platform hardware acceleration via llama.cpp (currently Vulkan and Metal are enabled)
  - Tested on: MacOS (M1 Max) and Linux (AMD RX 6700 XT).
- Easy to install and use without complex dependency chains
- Easy to include in shell scripts.
  - [`scripts/pageocr-screenshot`](scripts/pageocr-screenshot) interactive screenshot OCR for macOS and Linux/Wayland

## Install

```
cargo install https://github.com/DorianRudolph/pageocr
```

Note: Models and pdfium library will be downloaded at runtime.

## Key Dependencies

- [`LightOnOCR`](https://huggingface.co/lightonai/LightOnOCR-2-1B), the OCR model family used by this CLI
- [`llama-cpp-rs`](https://github.com/utilityai/llama-cpp-rs) for Rust bindings to GGUF inference
  - Currently using [my fork](https://github.com/DorianRudolph/llama-cpp-rs.git) backporting my LightOnOCR [fix](https://github.com/ggml-org/llama.cpp/pull/20877) to llama.cpp
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) for GGUF inference and MTMD multimodal support
- [`hf-hub`](https://crates.io/crates/hf-hub) for on-demand Hugging Face cache resolution
- [`pdfium-render`](https://crates.io/crates/pdfium-render) and [`pdfium-auto`](https://crates.io/crates/pdfium-auto) for PDF loading and rendering
- [`image`](https://crates.io/crates/image) for image decoding and resizing
- [`minijinja`](https://crates.io/crates/minijinja) for optional custom output templates

## Examples

### 1. OCR a scanned math page

Input:
- [`tests/fixtures/calculus_made_easy_page272.png`](tests/fixtures/calculus_made_easy_page272.png)

Output:
- [`examples/calculus_made_easy_page272_output.md`](examples/calculus_made_easy_page272_output.md)

Command:

```sh
pageocr tests/fixtures/calculus_made_easy_page272.png \
  --output examples/calculus_made_easy_page272_output.md
```

### 2. Full bbox OCR with extracted figures

Input:
- [`tests/fixtures/openstax_university_physics_selected_pages.pdf`](tests/fixtures/openstax_university_physics_selected_pages.pdf)

Output:
- [`examples/openstax_bbox/output.md`](examples/openstax_bbox/output.md)
- [`examples/openstax_bbox/images/`](examples/openstax_bbox/images/)

Command:

```sh
mkdir -p examples/openstax_bbox/images

pageocr \
  --variant bbox \
  tests/fixtures/openstax_university_physics_selected_pages.pdf \
  --output examples/openstax_bbox/output.md \
  --extract-images-dir examples/openstax_bbox/images
```

### 3. Interactive screenshot to clipboard

[`scripts/pageocr-screenshot`](scripts/pageocr-screenshot) captures an interactive screenshot,
runs `pageocr` on the image, copies the OCR result to the clipboard, and shows a short
success notification.

Platform support:
- macOS: uses `screencapture`, `pbcopy`, and `osascript`
- Linux/Wayland: uses `grim`, `slurp`, `wl-copy`, and optionally `notify-send`

The script prepends `../target/release` relative to its own location to `PATH`, so a local
`target/release/pageocr` build is used first when present; otherwise it falls back to a
globally installed `pageocr`.

Usage:

```sh
./scripts/pageocr-screenshot
```

On macOS, you can use Shortcuts.app to add a keybinding.
Note: Shortcuts.app has to be added to "Screen & System Audio Recording".

## Fixtures

Source attribution for the checked-in test fixtures is documented in
[`tests/fixtures/README.md`](tests/fixtures/README.md).
