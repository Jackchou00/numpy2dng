"""
Example: Generate DNG file from numpy array data
"""

import numpy as np
import os

from numpy2dng.core import RAW2DNG
from numpy2dng.dng import DNGTags, Tag
from numpy2dng.defs import (
    CFAPattern,
    PhotometricInterpretation,
    CalibrationIlluminant,
)


def create_dng():
    raw_data = np.ones((3000, 4000), dtype=np.uint16) * (2**14 - 1)  # Example raw data

    height, width = raw_data.shape

    color_matrix = [
        # Row 1: X coefficients
        (16360618, 10000000),  # 1.6360618
        (-44861874, 100000000),  # -0.44861874
        (-1874430, 10000000),  # -0.187443
        # Row 2: Y coefficients
        (-115164764, 1000000000),  # -0.115164764
        (14889708, 10000000),  # 1.4889708
        (-37380603, 100000000),  # -0.37380603
        # Row 3: Z coefficients
        (-4740228, 1000000000),  # -0.004740228
        (-32360125, 100000000),  # -0.32360125
        (13283415, 10000000),  # 1.3283415
    ]

    # White Balance: AsShotNeutral
    wb_r = (5022, 10000)  # 0.5022
    wb_g = (1, 1)  # 1.0
    wb_b = (5571, 10000)  # 0.5571

    tags = DNGTags()

    # ImageWidth / ImageLength: Dimensions of the main image
    tags.set(Tag.ImageWidth, width)
    tags.set(Tag.ImageLength, height)

    # BitsPerSample: Bit depth per pixel
    # 14-bit allows values 0-16383
    tags.set(Tag.BitsPerSample, 14)

    # SamplesPerPixel: Number of samples per pixel
    # 1 for Bayer CFA (Color Filter Array)
    tags.set(Tag.SamplesPerPixel, 1)

    # PhotometricInterpretation: Color space interpretation
    # 32803 = Color Filter Array (Bayer pattern)
    tags.set(
        Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array
    )

    # RowsPerStrip: Number of rows in each strip
    # Using full height for single strip
    tags.set(Tag.RowsPerStrip, height)

    # CFARepeatPatternDim: Dimensions of CFA pattern
    # [2, 2] means 2x2 repeating pattern
    tags.set(Tag.CFARepeatPatternDim, [2, 2])
    tags.set(Tag.CFAPattern, CFAPattern.RGGB)

    # ColorMatrix1: Color matrix for CalibrationIlluminant1
    tags.set(Tag.ColorMatrix1, color_matrix)

    # ColorMatrix2: Color matrix for CalibrationIlluminant2
    # Same as ColorMatrix1 in this example
    tags.set(Tag.ColorMatrix2, color_matrix)

    # CalibrationIlluminant1: Light source for ColorMatrix1
    tags.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.Standard_Light_A)

    # CalibrationIlluminant2: Light source for ColorMatrix2
    tags.set(Tag.CalibrationIlluminant2, CalibrationIlluminant.D65)

    # AsShotNeutral: White balance coefficients
    tags.set(Tag.AsShotNeutral, [wb_r, wb_g, wb_b])

    tags.set(Tag.BlackLevel, [512])
    tags.set(Tag.WhiteLevel, [16383])

    tags.set(Tag.XResolution, [(72, 1)])

    # Create converter instance
    converter = RAW2DNG()

    # Set options: tags and output directory
    converter.options(tags, os.path.dirname(__file__))

    output_path = converter.convert(raw_data, "example_output.dng")

    if os.path.exists(output_path):
        return True
    else:
        return False


if __name__ == "__main__":
    success = create_dng()
