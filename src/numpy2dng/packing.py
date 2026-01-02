import numpy as np
import math


def pack_raw_safe(data: np.ndarray, bit_depth: int) -> np.ndarray:
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
    out = np.zeros((data.shape[0], int(data.shape[1] * (1.25))), dtype=np.uint8)
    out[:, ::5] = data[:, ::4] >> 2
    out[:, 1::5] = (data[:, ::4] & 0b0000000000000011) << 6
    out[:, 1::5] += data[:, 1::4] >> 4
    out[:, 2::5] = (data[:, 1::4] & 0b0000000000001111) << 4
    out[:, 2::5] += data[:, 2::4] >> 6
    out[:, 3::5] = (data[:, 2::4] & 0b0000000000111111) << 2
    out[:, 3::5] += data[:, 3::4] >> 8
    out[:, 4::5] = data[:, 3::4] & 0b0000000011111111
    return out


def pack12(data: np.ndarray) -> np.ndarray:
    out = np.zeros((data.shape[0], int(data.shape[1] * (1.5))), dtype=np.uint8)
    out[:, ::3] = data[:, ::2] >> 4
    out[:, 1::3] = (data[:, ::2] & 0b0000000000001111) << 4
    out[:, 1::3] += data[:, 1::2] >> 8
    out[:, 2::3] = data[:, 1::2] & 0b0000001111111111
    return out


def pack14(data: np.ndarray) -> np.ndarray:
    out = np.zeros((data.shape[0], int(data.shape[1] * (1.75))), dtype=np.uint8)
    out[:, ::7] = data[:, ::4] >> 6
    out[:, 1::7] = (data[:, ::4] & 0b0000000000111111) << 2
    out[:, 1::7] += data[:, 1::4] >> 12
    out[:, 2::7] = data[:, 1::4] >> 4
    out[:, 2::7] &= 0xFF
    out[:, 3::7] = (data[:, 1::4] & 0b0000000000001111) << 4
    out[:, 3::7] += data[:, 2::4] >> 10
    out[:, 4::7] = data[:, 2::4] >> 2
    out[:, 4::7] &= 0xFF
    out[:, 5::7] = (data[:, 2::4] & 0b0000000000000011) << 6
    out[:, 5::7] += data[:, 3::4] >> 8
    out[:, 6::7] = data[:, 3::4] & 0b0000000011111111
    return out
