# numpy2dng

Pure Python utility for creating Adobe DNG files from RAW image data stored in NumPy arrays.

This project is a refactor of the original PiDNG library, focused on a smaller, cleaner core and modern packaging.

## Requirements

- Python 3.10+
- NumPy 2.0+

## Features

- Pure Python (no C extensions)
- Uncompressed DNG output
- Supports 8/10/12/14/16-bit integer RAW (via packing for 10/12/14-bit)
- Supports `float32` RAW (writes DNG 1.4-compatible files)
- “Safe packing” for 10/12/14-bit data with any image width (no special alignment required)

## Non-goals / Limitations

- Not a full RAW pipeline (no demosaic, no color processing)
- Does not auto-detect camera metadata; you must provide the required DNG/TIFF tags
- No JPEG/LJ92 compression (uncompressed only)

## Installation

Using `uv` (recommended):

```bash
uv add numpy2dng
```

Using `pip`:

```bash
pip install numpy2dng
```

## Usage

Typical flow:

1. Prepare a 2D NumPy array of shape `(height, width)` with dtype `uint16` (or `float32`).
2. Build a `numpy2dng.dng.DNGTags` instance and set at least:
   - `Tag.ImageWidth`, `Tag.ImageLength`, `Tag.BitsPerSample`
3. Convert and write a `.dng` using `numpy2dng.core.RAW2DNG`.

For performance-sensitive workflows, `RAW2DNG.convert(..., file=...)` can write directly to an open binary file handle (avoids an extra copy).

## API Overview

- `numpy2dng.core.RAW2DNG`: main converter for NumPy arrays → DNG
- `numpy2dng.dng.DNGTags` / `numpy2dng.dng.Tag`: tag container and tag definitions

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Credits

Based on (and inspired by) [PiDNG](https://github.com/schoolpost/PiDNG)

## Future work

- Add more metadata tags and convenience methods for common camera models
- Support for additional data types and compression methods
- Improve documentation and add usage examples

## License

Same as PiDNG: MIT License. See `LICENSE` file for details.
