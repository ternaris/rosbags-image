# Copyright 2019-2022  Ternaris.
# SPDX-License-Identifier: Apache-2.0
"""Image conversion between ROS and OpenCV formats."""

from __future__ import annotations

import re
import sys
from contextlib import suppress
from enum import IntEnum
from itertools import product
from typing import TYPE_CHECKING

import cv2  # type: ignore
import numpy
from rosbags.typesys.types import sensor_msgs__msg__CompressedImage as CompressedImage

if TYPE_CHECKING:
    from typing import Any, Optional, Union

    if sys.version_info < (3, 9):
        Imagebytes = numpy.ndarray[Any, Any]
    else:
        from numpy.typing import NDArray
        Imagebytes = NDArray[Any]
    from rosbags.typesys.types import sensor_msgs__msg__Image as Image

ENC_RE = re.compile('(8U|8S|16U|16S|32S|32F|64F)(?:C([0-9]+))?')
DIGITS_RE = re.compile(r'^\d+')


class ImageError(Exception):
    """Unsupported Image Format."""


class ImageFormatError(ImageError):
    """Unsupported Image Format."""


class ImageConversionError(ImageError):
    """Unsupported Image Conversion."""


class Format(IntEnum):
    """Supported Image Formats."""

    GENERIC = -1
    GRAY = 0
    RGB = 1
    BGR = 2
    RGBA = 3
    BGRA = 4
    YUV = 5
    BAYER_RG = 6
    BAYER_BG = 7
    BAYER_GB = 8
    BAYER_GR = 9


CONVERSIONS = {
    key: getattr(cv2, f'COLOR_{n1}2{n2}{({"YUV": "_Y422"}).get(n1, "")}', None)
    for key in product(*(2 * (list(Format)[1:],)))
    if (n1 := key[0].name) == (n2 := key[1].name) or hasattr(cv2, f'COLOR_{n1}2{n2}')
}

DEPTHMAP = {
    '8U': 'uint8',
    '8S': 'int8',
    '16U': 'uint16',
    '16S': 'int16',
    '32S': 'int32',
    '32F': 'float32',
    '64F': 'float64',
}

ENCODINGMAP = {
    'mono8': (8, Format.GRAY, 'uint8', 1),
    'bayer_rggb8': (8, Format.BAYER_BG, 'uint8', 1),
    'bayer_bggr8': (8, Format.BAYER_RG, 'uint8', 1),
    'bayer_gbrg8': (8, Format.BAYER_GR, 'uint8', 1),
    'bayer_grbg8': (8, Format.BAYER_GB, 'uint8', 1),
    'yuv422': (8, Format.YUV, 'uint8', 2),
    'bgr8': (8, Format.BGR, 'uint8', 3),
    'rgb8': (8, Format.RGB, 'uint8', 3),
    'bgra8': (8, Format.BGRA, 'uint8', 4),
    'rgba8': (8, Format.RGBA, 'uint8', 4),
    'mono16': (16, Format.GRAY, 'uint16', 1),
    'bayer_rggb16': (16, Format.BAYER_BG, 'uint16', 1),
    'bayer_bggr16': (16, Format.BAYER_RG, 'uint16', 1),
    'bayer_gbrg16': (16, Format.BAYER_GR, 'uint16', 1),
    'bayer_grbg16': (16, Format.BAYER_GB, 'uint16', 1),
    'bgr16': (16, Format.BGR, 'uint16', 3),
    'rgb16': (16, Format.RGB, 'uint16', 3),
    'bgra16': (16, Format.BGRA, 'uint16', 4),
    'rgba16': (16, Format.RGBA, 'uint16', 4),
}


def to_cvtype(encoding: str) -> tuple[int, Format, str, int]:
    """Get typeinfo for encoding.

    Args:
        encoding: String representation of image encoding.

    Returns:
        Tuple describing OpenCV type.

    Raises:
        ImageFormatError: If encoding cannot be parsed.

    """
    with suppress(KeyError):
        return ENCODINGMAP[encoding]

    if mat := ENC_RE.fullmatch(encoding):
        depth, nchan = mat.groups()
        mat = DIGITS_RE.search(depth)
        assert mat
        return (int(mat.group()), Format.GENERIC, DEPTHMAP[depth], int(nchan or 1))

    raise ImageFormatError(f'Format {encoding!r} is not supported')


def convert_color(src: Imagebytes, src_color_space: str, dst_color_space: str) -> Imagebytes:
    """Convert color space.

    Args:
        src: Source image.
        src_color_space: Source color space.
        dst_color_space: Destination color space.

    Returns:
        Image in destination color space, or source image if conversion is
        a noop.

    Raises:
        ImageConversionError: If conversion is not supported.

    """
    if src_color_space == dst_color_space:
        return src

    src_depth, src_fmt, src_typestr, src_nchan = to_cvtype(src_color_space)
    dst_depth, dst_fmt, dst_typestr, dst_nchan = to_cvtype(dst_color_space)

    try:
        conversion = CONVERSIONS[(src_fmt, dst_fmt)]
    except KeyError:
        if Format.GENERIC not in (src_fmt, dst_fmt) or src_nchan != dst_nchan:
            raise ImageConversionError(
                f'Conversion {src_color_space!r} -> {dst_color_space!r} is not supported',
            ) from None
        conversion = None

    if conversion:
        src = cv2.cvtColor(src, conversion)  # pyright: ignore

    if src_typestr != dst_typestr:
        if src_depth == 8 and dst_depth == 16:
            return numpy.multiply(  # type: ignore
                src.astype(dst_typestr, copy=False),
                65535. / 255.,
            )
        if src_depth == 16 and dst_depth == 8:
            return numpy.multiply(  # type: ignore
                src,
                255. / 65535.,
            ).astype(dst_typestr, copy=False)
        return src.astype(dst_typestr, copy=False)
    return src


def image_to_cvimage(msg: Image, color_space: Optional[str] = None) -> Imagebytes:
    """Convert sensor_msg/msg/Image to OpenCV image.

    Args:
        msg: Image message.
        color_space: Color space of output image.

    Returns:
        OpenCV image.

    """
    _, _, typestr, nchan = to_cvtype(msg.encoding)
    shape = (msg.height, msg.width) if nchan == 1 else (msg.height, msg.width, nchan)
    dtype = numpy.dtype(typestr)  # .newbyteorder('>' if msg.is_bigendian else '<')
    img: Imagebytes = numpy.ndarray(shape=shape, dtype=dtype, buffer=msg.data)
    if msg.is_bigendian != (sys.byteorder != 'little'):
        img.byteswap(inplace=True)

    if color_space:
        return convert_color(img, msg.encoding, color_space)
    return img


def compressed_image_to_cvimage(
    msg: CompressedImage,
    color_space: Optional[str] = None,
) -> Imagebytes:
    """Convert sensor_msg/msg/CompressedImage to OpenCV image.

    Args:
        msg: CompressedImage message.
        color_space: Color space of output image.

    Returns:
        OpenCV image.

    """
    img: Imagebytes = cv2.imdecode(  # pyright: ignore
        numpy.frombuffer(msg.data, numpy.uint8),
        cv2.IMREAD_ANYCOLOR,  # pyright: ignore
    )
    if color_space:
        return convert_color(img, 'bgr8', color_space)
    return img


def message_to_cvimage(
    msg: Union[CompressedImage, Image],
    color_space: Optional[str] = None,
) -> Imagebytes:
    """Convert ROS message to OpenCV image.

    Args:
        msg: CompressedImage or Image message.
        color_space: Color space of output image.

    Returns:
        OpenCV image.

    """
    if isinstance(msg, CompressedImage):
        return compressed_image_to_cvimage(msg, color_space)
    return image_to_cvimage(msg, color_space)
