"""Pixel packing utilities for common RAW bit depths."""

import numpy as np
import math


def pack_raw_safe(data: np.ndarray, bit_depth: int) -> np.ndarray:
    """Pack a 2D uint16 RAW array into a byte-aligned packed format.

    Args:
        data: Input array with shape `(height, width)`.
        bit_depth: Bit depth per pixel (supported: 10, 12, 14).

    Returns:
        Packed bytes as a `uint8` NumPy array with shape `(height, packed_width)`.
    """
    h, w = data.shape
    if bit_depth == 10:
        group_size = 4
        pack_func = pack10
    elif bit_depth == 12:
        group_size = 2
        pack_func = pack12
    elif bit_depth == 14:
        group_size = 4
        pack_func = pack14
    else:
        raise ValueError("Unsupported bit depth")
    remainder = w % group_size
    pad_width = 0
    data_padded = data
    if remainder != 0:
        pad_width = group_size - remainder
        # ((top, bottom), (left, right))
        data_padded = np.pad(
            data, ((0, 0), (0, pad_width)), "constant", constant_values=0
        )
    out_padded = pack_func(data_padded)
    valid_byte_width = math.ceil(w * bit_depth / 8)
    out = out_padded[:, :valid_byte_width]
    return out


def pack10(data: np.ndarray) -> np.ndarray:
    """Pack 10-bit pixels into 5 bytes per 4 pixels.
    
    Args:
        data: Input array with shape `(height, width)`.

    Returns:
        Packed bytes as a `uint8` NumPy array.
    """
    out = np.zeros((data.shape[0], data.shape[1] * 5 // 4), dtype=np.uint8)
    out[:, 0::5] = data[:, 0::4] >> 2
    out[:, 1::5] = ((data[:, 0::4] & 0x03) << 6) | (data[:, 1::4] >> 4)
    out[:, 2::5] = ((data[:, 1::4] & 0x0F) << 4) | (data[:, 2::4] >> 6)
    out[:, 3::5] = ((data[:, 2::4] & 0x3F) << 2) | (data[:, 3::4] >> 8)
    out[:, 4::5] = data[:, 3::4] & 0xFF
    return out


def pack12(data: np.ndarray) -> np.ndarray:
    """Pack 12-bit pixels into 3 bytes per 2 pixels.

    Args:
        data: Input array with shape `(height, width)`.

    Returns:
        Packed bytes as a `uint8` NumPy array.
    """
    out = np.zeros((data.shape[0], int(data.shape[1] * 3 // 2)), dtype=np.uint8)
    out[:, ::3] = (data[:, ::2] & 0x0FF0) >> 4
    out[:, 1::3] = (data[:, ::2] & 0x000F) << 4 | (data[:, 1::2] & 0x0F00) >> 8
    out[:, 2::3] = data[:, 1::2] & 0x00FF
    return out


def pack14(data: np.ndarray) -> np.ndarray:
    """Pack 14-bit pixels into 7 bytes per 4 pixels.

    Args:
        data: Input array with shape `(height, width)`.

    Returns:
        Packed bytes as a `uint8` NumPy array.
    """
    out = np.zeros((data.shape[0], data.shape[1] * 7 // 4), dtype=np.uint8)
    out[:, 0::7] = data[:, 0::4] >> 6
    out[:, 1::7] = ((data[:, 0::4] & 0x3F) << 2) | (data[:, 1::4] >> 12)
    out[:, 2::7] = (data[:, 1::4] >> 4) & 0xFF
    out[:, 3::7] = ((data[:, 1::4] & 0x0F) << 4) | (data[:, 2::4] >> 10)
    out[:, 4::7] = (data[:, 2::4] >> 2) & 0xFF
    out[:, 5::7] = ((data[:, 2::4] & 0x03) << 6) | (data[:, 3::4] >> 8)
    out[:, 6::7] = data[:, 3::4] & 0xFF
    return out
