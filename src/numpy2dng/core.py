"""Core conversion helpers for writing NumPy arrays to DNG."""

import os
import numpy as np
import types
from typing import BinaryIO
from .dng import Tag, dngIFD, dngTag, DNG, DNGTags
from .defs import Compression, DNGVersion, SampleFormat
from .packing import pack_raw_safe


class DNGBASE:
    """Base converter for writing NumPy RAW frames into a DNG container."""

    def __init__(self) -> None:
        """Initialize the converter with no options configured."""
        self.path = None
        self.tags = None
        self.filter = None

    def __data_condition__(self, data: np.ndarray) -> None:
        """Validate that the input frame uses a supported dtype.

        Args:
            data: Input image as a NumPy array.

        Returns:
            None
        """
        if data.dtype != np.uint16 and data.dtype != np.float32:
            raise Exception(
                "RAW Data is not in correct format. Must be uint16_t or float32_t Numpy Array. "
            )

    def __tags_condition__(self, tags: DNGTags) -> None:
        """Validate that required DNG tags are present.

        Args:
            tags: DNG tag container to validate.

        Returns:
            None
        """
        if not tags.get(Tag.ImageWidth):
            raise Exception("No width is defined in tags.")
        if not tags.get(Tag.ImageLength):
            raise Exception("No height is defined in tags.")
        if not tags.get(Tag.BitsPerSample):
            raise Exception("Bit per pixel is not defined.")

    def __unpack_pixels__(self, data: np.ndarray) -> np.ndarray:
        """Transform pixel layout before filtering/packing, if needed.

        Args:
            data: Input image as a NumPy array.

        Returns:
            Unpacked pixel array.
        """
        return data

    def __filter__(
        self, rawFrame: np.ndarray, filter: types.FunctionType
    ) -> np.ndarray:
        """Apply an optional user filter to a RAW frame.

        Args:
            rawFrame: Input frame as a NumPy array.
            filter: Callable that accepts and returns a NumPy array, or `None`.

        Returns:
            The filtered frame as a NumPy array.
        """
        if not filter:
            return rawFrame

        processed = filter(rawFrame)
        if not isinstance(processed, np.ndarray):
            raise TypeError("return value is not a valid numpy array!")
        elif processed.shape != rawFrame.shape:
            raise ValueError("return array does not have the same shape!")
        if processed.dtype != np.uint16:
            raise ValueError("array data type is invalid!")

        return processed

    def __process__(
        self, rawFrame: np.ndarray, tags: DNGTags, file: BinaryIO = None
    ) -> bytearray:
        """Pack pixels and assemble a DNG byte buffer (and optionally write it).

        Args:
            rawFrame: Frame data with shape `(height, width, ...)`.
            tags: DNG tag container describing image geometry and encoding.
            file: Optional writable binary file-like object to stream output into.

        Returns:
            DNG file contents as a bytearray when `file` is not provided; otherwise a
            bytearray containing the header/tag buffer.
        """
        width = tags.get(Tag.ImageWidth).rawValue[0]
        length = tags.get(Tag.ImageLength).rawValue[0]
        bpp = tags.get(Tag.BitsPerSample).rawValue[0]

        if rawFrame.ndim < 2:
            raise ValueError(
                f"rawFrame must be at least 2D (height, width); got shape {rawFrame.shape!r}"
            )

        frame_length, frame_width = rawFrame.shape[:2]
        if frame_width != width or frame_length != length:
            raise ValueError(
                "rawFrame shape does not match DNG tags: "
                f"got {(frame_length, frame_width)}, expected {(length, width)}"
            )

        # Pure Python implementation - always uncompressed
        compression_scheme = Compression.Uncompressed

        sample_format = SampleFormat.Uint
        backward_version = DNGVersion.V1_0
        if rawFrame.dtype == np.float32:
            sample_format = SampleFormat.FloatingPoint
            # Floating-point data requires DNG 1.4
            backward_version = DNGVersion.V1_4

        # Pure Python packing (no C extension required)
        if bpp == 8:
            packedFrame = rawFrame.astype("uint8")
        elif bpp == 10:
            packedFrame = pack_raw_safe(rawFrame, 10)
        elif bpp == 12:
            packedFrame = pack_raw_safe(rawFrame, 12)
        elif bpp == 14:
            packedFrame = pack_raw_safe(rawFrame, 14)
        else:
            # 16-bit integers or 32-bit floats
            packedFrame = rawFrame
        # These buffers are all contiguous, so the optimised output route
        # can use the underlying memoryview, no need to convert to bytes.
        tile = packedFrame.data if file else packedFrame.tobytes()

        dngTemplate = DNG()

        dngTemplate.ImageDataStrips.append(tile)
        # set up the FULL IFD
        mainIFD = dngIFD()
        mainTagStripOffset = dngTag(
            Tag.StripOffsets, [0 for tile in dngTemplate.ImageDataStrips]
        )
        mainIFD.tags.append(mainTagStripOffset)
        mainIFD.tags.append(dngTag(Tag.NewSubfileType, [0]))
        mainIFD.tags.append(
            dngTag(
                Tag.StripByteCounts, [len(tile) for tile in dngTemplate.ImageDataStrips]
            )
        )
        mainIFD.tags.append(dngTag(Tag.Compression, [compression_scheme]))
        mainIFD.tags.append(dngTag(Tag.Software, "PiDNG"))
        mainIFD.tags.append(dngTag(Tag.DNGVersion, DNGVersion.V1_4))
        mainIFD.tags.append(dngTag(Tag.DNGBackwardVersion, backward_version))
        mainIFD.tags.append(dngTag(Tag.SampleFormat, [sample_format]))

        for tag in tags.list():
            try:
                mainIFD.tags.append(tag)
            except Exception as e:
                print("TAG Encoding Error!", e, tag)

        dngTemplate.IFDs.append(mainIFD)

        totalLength = dngTemplate.dataLen()

        mainTagStripOffset.setValue(
            [k for offset, k in dngTemplate.StripOffsets.items()]
        )

        buf = bytearray(totalLength)
        dngTemplate.setBuffer(buf)
        # The file parameter will cause the optimised output route to be used,
        # where appropriate.
        dngTemplate.write(file=file)

        return buf

    def options(self, tags: DNGTags, path: str) -> None:
        """Configure output options for subsequent conversions.

        Args:
            tags: DNG tag container used for generated files.
            path: Output directory used when `convert` writes to disk.

        Returns:
            None
        """
        self.__tags_condition__(tags)
        self.tags = tags
        self.path = path

    def convert(self, image: np.ndarray, filename="", file: BinaryIO = None):
        """Convert a NumPy image into a DNG file or in-memory buffer.

        Args:
            image: Input image as a NumPy array.
            filename: Output filename (without or with `.dng`) when writing to disk.
            file: Optional open binary file handle (or `io.BytesIO`) to stream output.

        Returns:
            If `file` is provided, returns `None`. If `filename` is provided, returns
            the output file path. Otherwise returns the DNG bytes.
        """
        # The file parameter can be passed an open file handle, or a BytesIO,
        # and this function will take an optimised route to writing the output.
        # Note that the pixel data is not copied to self.buf (which is why it's
        # faster) in this case.

        if self.tags is None:
            raise Exception("Options have not been set!")

        # valdify incoming data
        self.__data_condition__(image)
        unpacked = self.__unpack_pixels__(image)
        filtered = self.__filter__(unpacked, self.filter)
        buf = self.__process__(filtered, self.tags, file=file)

        if file:
            # For the optimised output route, __process__ has already written
            # the output for us, so we are finished.
            return

        file_output = False
        if len(filename) > 0:
            file_output = True

        if file_output:
            if not filename.endswith(".dng"):
                filename = filename + ".dng"
            outputDNG = os.path.join(self.path, filename)
            with open(outputDNG, "wb") as outfile:
                outfile.write(buf)
            return outputDNG
        else:
            return buf


class RAW2DNG(DNGBASE):
    """Converter for writing raw NumPy frames into DNG."""

    def __init__(self) -> None:
        """Initialize the converter.

        Args:
            None

        Returns:
            None
        """
        super().__init__()


class CAM2DNG(DNGBASE):
    """Converter that sources its tags from a camera model object."""

    def __init__(self, model) -> None:
        """Initialize the converter with a camera model.

        Args:
            model: Camera model object exposing a `tags` attribute.

        Returns:
            None
        """
        super().__init__()
        self.model = model

    def options(self, path: str) -> None:
        """Configure output directory using tags from the configured model.

        Args:
            path: Output directory used when `convert` writes to disk.

        Returns:
            None
        """
        self.__tags_condition__(self.model.tags)
        self.tags = self.model.tags
        self.path = path
