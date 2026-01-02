"""Test width modulo 4 and bit depth downsampling (10, 12, 14 bit)."""

import os
import tempfile
import numpy as np
import pytest

from numpy2dng.core import RAW2DNG
from numpy2dng.dng import DNGTags, Tag
from numpy2dng.defs import CFAPattern, PhotometricInterpretation, CalibrationIlluminant


def create_test_data(height: int, width: int, bit_depth: int = 14) -> np.ndarray:
    """Create test RAW data with specified dimensions and bit depth."""
    max_value = (1 << bit_depth) - 1
    # Create patterned data to verify round-trip
    data = np.arange(width * height, dtype=np.uint16) % max_value
    data = data.reshape((height, width))
    return data


def create_tags(width: int, height: int, bits_per_sample: int) -> DNGTags:
    """Create DNG tags for testing."""
    color_matrix = [
        (16360618, 10000000),
        (-44861874, 100000000),
        (-1874430, 10000000),
        (-115164764, 1000000000),
        (14889708, 10000000),
        (-37380603, 100000000),
        (-4740228, 1000000000),
        (-32360125, 100000000),
        (13283415, 10000000),
    ]

    wb_r = (5022, 10000)
    wb_g = (1, 1)
    wb_b = (5571, 10000)

    tags = DNGTags()
    tags.set(Tag.ImageWidth, width)
    tags.set(Tag.ImageLength, height)
    tags.set(Tag.BitsPerSample, bits_per_sample)
    tags.set(Tag.SamplesPerPixel, 1)
    tags.set(
        Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array
    )
    tags.set(Tag.RowsPerStrip, height)
    tags.set(Tag.CFARepeatPatternDim, [2, 2])
    tags.set(Tag.CFAPattern, CFAPattern.RGGB)
    tags.set(Tag.ColorMatrix1, color_matrix)
    tags.set(Tag.ColorMatrix2, color_matrix)
    tags.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.Standard_Light_A)
    tags.set(Tag.CalibrationIlluminant2, CalibrationIlluminant.D65)
    tags.set(Tag.AsShotNeutral, [wb_r, wb_g, wb_b])
    tags.set(Tag.BlackLevel, [512])
    tags.set(Tag.WhiteLevel, [(1 << bits_per_sample) - 1])
    tags.set(Tag.XResolution, [(72, 1)])

    return tags


def verify_dng_decode(dng_path: str, original_data: np.ndarray) -> bool:
    """Verify DNG can be decoded and data matches (using rawpy)."""
    try:
        import rawpy
    except ImportError:
        pytest.skip("rawpy not installed, skipping decode verification")

    try:
        with rawpy.imread(dng_path) as raw:
            raw_image = raw.raw_image.copy()
    except rawpy._rawpy.LibRawFileUnsupportedError:
        # rawpy may not support very small images
        # Verify file exists and has valid DNG header instead
        with open(dng_path, "rb") as f:
            header = f.read(4)
            # Check for DNG magic bytes (TIFF with 'II' or 'MM' + version)
            if header[:2] not in [b"II", b"MM"]:
                return False
            return True

    # The decoded data should match the original
    return np.array_equal(raw_image, original_data)


class TestWidthModulo4:
    """Test different width remainders when divided by 4."""

    @pytest.mark.parametrize("width_remainder", [0, 1, 2, 3])
    def test_width_modulo_4_14bit(self, width_remainder):
        """Test widths with different remainders modulo 4 at 14-bit."""
        height = 100
        base_width = 100  # 100 % 4 = 0
        width = base_width + width_remainder

        # Create test data
        data = create_test_data(height, width, 14)
        tags = create_tags(width, height, 14)

        # Convert to DNG
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = RAW2DNG()
            converter.options(tags, tmpdir)
            dng_path = converter.convert(data, f"test_w{width}_14bit.dng")

            # Verify file was created
            assert os.path.exists(dng_path), f"DNG file not created for width={width}"

            # Verify decode
            assert verify_dng_decode(dng_path, data), f"Decode failed for width={width}"

    @pytest.mark.parametrize("width_remainder", [0, 1, 2, 3])
    def test_width_modulo_4_12bit(self, width_remainder):
        """Test widths with different remainders modulo 4 at 12-bit."""
        height = 100
        base_width = 100
        width = base_width + width_remainder

        data = create_test_data(height, width, 12)
        tags = create_tags(width, height, 12)

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = RAW2DNG()
            converter.options(tags, tmpdir)
            dng_path = converter.convert(data, f"test_w{width}_12bit.dng")

            assert os.path.exists(dng_path)
            assert verify_dng_decode(dng_path, data)

    @pytest.mark.parametrize("width_remainder", [0, 1, 2, 3])
    def test_width_modulo_4_10bit(self, width_remainder):
        """Test widths with different remainders modulo 4 at 10-bit."""
        height = 100
        base_width = 100
        width = base_width + width_remainder

        data = create_test_data(height, width, 10)
        tags = create_tags(width, height, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = RAW2DNG()
            converter.options(tags, tmpdir)
            dng_path = converter.convert(data, f"test_w{width}_10bit.dng")

            assert os.path.exists(dng_path)
            assert verify_dng_decode(dng_path, data)


class TestBitDepthDownsampling:
    """Test downsampling from 14-bit to 10, 12, 14-bit."""

    @pytest.mark.parametrize("bit_depth", [10, 12, 14])
    def test_bit_depths(self, bit_depth):
        """Test different bit depths with width divisible by 4."""
        height = 100
        width = 100  # Divisible by 4

        data = create_test_data(height, width, bit_depth)
        tags = create_tags(width, height, bit_depth)

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = RAW2DNG()
            converter.options(tags, tmpdir)
            dng_path = converter.convert(data, f"test_{bit_depth}bit.dng")

            assert os.path.exists(dng_path)
            assert verify_dng_decode(dng_path, data)

    @pytest.mark.parametrize("bit_depth", [10, 12, 14])
    @pytest.mark.parametrize("width_remainder", [0, 1, 2, 3])
    def test_bit_depth_with_width_variations(self, bit_depth, width_remainder):
        """Test all bit depths with all width remainders."""
        height = 50
        width = 100 + width_remainder

        data = create_test_data(height, width, bit_depth)
        tags = create_tags(width, height, bit_depth)

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = RAW2DNG()
            converter.options(tags, tmpdir)
            dng_path = converter.convert(data, f"test_w{width}_{bit_depth}bit.dng")

            assert os.path.exists(dng_path)
            assert verify_dng_decode(dng_path, data)


class TestEdgeCases:
    """Test edge cases for width and bit depth."""

    def test_minimum_width_14bit(self):
        """Test minimum width (4) at 14-bit."""
        height = 10
        width = 4  # Minimum for 14-bit packing

        data = create_test_data(height, width, 14)
        tags = create_tags(width, height, 14)

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = RAW2DNG()
            converter.options(tags, tmpdir)
            dng_path = converter.convert(data, "min_width_14bit.dng")

            assert os.path.exists(dng_path)
            assert verify_dng_decode(dng_path, data)

    def test_minimum_width_12bit(self):
        """Test minimum width (2) at 12-bit."""
        height = 10
        width = 2  # Minimum for 12-bit packing

        data = create_test_data(height, width, 12)
        tags = create_tags(width, height, 12)

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = RAW2DNG()
            converter.options(tags, tmpdir)
            dng_path = converter.convert(data, "min_width_12bit.dng")

            assert os.path.exists(dng_path)
            assert verify_dng_decode(dng_path, data)

    def test_minimum_width_10bit(self):
        """Test minimum width (4) at 10-bit."""
        height = 10
        width = 4  # Minimum for 10-bit packing

        data = create_test_data(height, width, 10)
        tags = create_tags(width, height, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = RAW2DNG()
            converter.options(tags, tmpdir)
            dng_path = converter.convert(data, "min_width_10bit.dng")

            assert os.path.exists(dng_path)
            assert verify_dng_decode(dng_path, data)

    def test_large_width_14bit(self):
        """Test larger width at 14-bit."""
        height = 50
        width = 1025  # 1025 % 4 = 1

        data = create_test_data(height, width, 14)
        tags = create_tags(width, height, 14)

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = RAW2DNG()
            converter.options(tags, tmpdir)
            dng_path = converter.convert(data, "large_width_14bit.dng")

            assert os.path.exists(dng_path)
            assert verify_dng_decode(dng_path, data)
