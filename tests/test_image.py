# Copyright 2020  Ternaris.
# SPDX-License-Identifier: AGPL-3.0-only
"""Rosbags image tests."""

from __future__ import annotations

import struct
import sys
from base64 import b64decode
from itertools import product
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy
import pytest
from rosbags.typesys.types import builtin_interfaces__msg__Time as Time
from rosbags.typesys.types import sensor_msgs__msg__CompressedImage as CompressedImage
from rosbags.typesys.types import sensor_msgs__msg__Image as Image
from rosbags.typesys.types import std_msgs__msg__Header as Header

from rosbags.image import (
    ImageConversionError,
    ImageFormatError,
    compressed_image_to_cvimage,
    image_to_cvimage,
    message_to_cvimage,
)

if TYPE_CHECKING:
    from typing import Any, Union

FORMATS = [
    'rgb8',
    'rgba8',
    'rgb16',
    'rgba16',
    'bgr8',
    'bgra8',
    'bgr16',
    'bgra16',
    'mono8',
    'mono16',
    '8U',
    '8UC1',
    '8UC2',
    '8UC3',
    '8UC4',
    '8SC1',
    '8SC2',
    '8SC3',
    '8SC4',
    '16UC1',
    '16UC2',
    '16UC3',
    '16UC4',
    '16SC1',
    '16SC2',
    '16SC3',
    '16SC4',
    '32SC1',
    '32SC2',
    '32SC3',
    '32SC4',
    '32FC1',
    '32FC2',
    '32FC3',
    '32FC4',
    '64FC1',
    '64FC2',
    '64FC3',
    '64FC4',
    'bayer_rggb8',
    'bayer_bggr8',
    'bayer_gbrg8',
    'bayer_grbg8',
    'bayer_rggb16',
    'bayer_bggr16',
    'bayer_gbrg16',
    'bayer_grbg16',
    'yuv422',
]

HEADER = Header(Time(0, 0), '')


def get_desc(name: str) -> tuple[int, int, bool]:
    """Get format description from name.

    Args:
        name: Format name.

    Returns:
        Format description.

    Raises:
        ValueError: If encoding unknown.

    """
    if name[0:4] in ('rgba', 'bgra') or name.endswith('C4'):
        channels = 4
    elif name[0:3] in ('rgb', 'bgr') or name.endswith('C3'):
        channels = 3
    elif name.endswith('C2') or name == 'yuv422':
        channels = 2
    elif (  # noqa: SIM106
        name.startswith('mono') or name.startswith('bayer_') or name[-1] in 'USF1'
    ):
        channels = 1
    else:
        raise ValueError(f'Unexpected encoding {name}')

    if name.startswith('8') or name.endswith('8') or name == 'yuv422':
        bits = 8
    elif name.startswith('16') or name.endswith('16'):
        bits = 16
    elif name.startswith('32'):
        bits = 32
    elif name.startswith('64'):  # noqa: SIM106
        bits = 64
    else:
        raise ValueError(f'Unexpected encoding {name}')

    return (channels, bits, 'FC' in name)


def generate_image(
    fmt: str,
    is_big: bool = sys.byteorder != 'little',
) -> tuple[Image, int, Union[int, numpy.ndarray[Any, numpy.dtype[numpy.uint8]]]]:
    """Generate ROS image message for specific format.

    Args:
        fmt: Image encoding.
        is_big: Is byteorder big endian.

    Returns:
        Image message, bitsize, value.

    """
    channels, bits, is_float = get_desc(fmt)
    width, height = 32, 24

    pxsize = channels * bits >> 3
    step = width * pxsize

    # create empty image and set the first channel of one singular pixel to 0.25 of max value
    data = bytearray(height * step)
    pxpos = step * int(height / 2 - 1) + pxsize * int(width / 2 - 1)
    ftype = {True: {32: 'f', 64: 'd'}, False: {8: 'B', 16: 'H', 32: 'I', 64: 'Q'}}[is_float][bits]
    cval = (1 << bits - 2)
    data[pxpos:pxpos + bits >> 8] = struct.pack(f'{">" if is_big else "<"}{ftype}', cval)

    # set remaining channels to 0, tests use this source value to check encoding conversions
    val = cval if channels == 1 else numpy.array([cval] + [0] * (channels - 1))

    return Image(
        header=HEADER,
        height=height,
        width=width,
        encoding=fmt,
        is_bigendian=is_big,
        step=step,
        data=numpy.frombuffer(data, numpy.uint8),  # type: ignore
    ), bits, val


def test_badencoding() -> None:
    """Test image message with bad encoding."""
    image, _, _ = generate_image('mono8')
    image.encoding = 'bayer'
    with pytest.raises(ImageFormatError, match='bayer'):
        image_to_cvimage(image)


@pytest.mark.parametrize('fmt', FORMATS)
def test_noconvert(fmt: str) -> None:
    """Test without color format conversion."""
    image, _, val = generate_image(fmt)
    img = image_to_cvimage(image, color_space=None)
    assert img.shape[0:2] == (image.height, image.width)
    numpy.testing.assert_array_equal(
        img[11:13, 15:17],
        [[val, val * 0], [val * 0, val * 0]],  # type: ignore
    )


@pytest.mark.parametrize('fmt', FORMATS)
def test_convert_mono(fmt: str) -> None:
    """Test with color format conversion to mono."""
    image, bits, val = generate_image(fmt)

    # unsupported auto conversions
    if fmt[-2:] in ('C2', 'C3', 'C4', '22'):
        with pytest.raises(ImageConversionError):
            image_to_cvimage(image, 'mono8')
        return

    img = image_to_cvimage(image, 'mono8')
    assert img.shape == (image.height, image.width)

    # apply CCIR 601
    if fmt.startswith('rgb'):
        assert isinstance(val, numpy.ndarray)
        val = int(val[0] * 0.2989)
    elif fmt.startswith('bgr'):
        assert isinstance(val, numpy.ndarray)
        val = int(val[0] * 0.1140)

    # 16 bits are autoscaled down, higher bits do not fit into uint8
    if bits == 16:
        val = int(val / 257)
    elif bits > 16:
        val = 0

    if fmt.startswith('bayer_rggb'):
        expect = [[7, 4 - (bits == 16)], [3, 2 - (bits == 16)]]
    elif fmt.startswith('bayer_bggr'):
        expect = [[19, 10 - (bits == 16)], [9, 5 - (bits == 16)]]
    elif fmt.startswith('bayer_g'):
        expect = [[37, 9], [9, 0]]
    else:
        assert isinstance(val, int)
        expect = [[val, val * 0], [val * 0, val * 0]]

    numpy.testing.assert_array_equal(img[11:13, 15:17], numpy.array(expect))


@pytest.mark.parametrize(('fmt', 'endian'), product(FORMATS, (False, True)))
def test_convert_bgr8(fmt: str, endian: bool) -> None:
    """Test with color format conversion to bgr8."""
    image, bits, val = generate_image(fmt, endian)

    # unsupported auto conversions
    if fmt[-2:] in ('C1', 'C2', 'C4') or fmt[-1] in 'USF':
        with pytest.raises(ImageConversionError):
            image_to_cvimage(image, 'bgr8')
        return

    img = image_to_cvimage(image, 'bgr8')
    assert img.shape == (image.height, image.width, 3)

    # flip rgb, remove alpha, expand mono
    if fmt.startswith('rgb'):
        val = numpy.flip(val[:3])  # type: ignore
    elif fmt.startswith('bgr'):
        assert isinstance(val, numpy.ndarray)
        val = val[:3]
    elif fmt.startswith('mono'):
        assert isinstance(val, int)
        val = numpy.array([val, val, val])

    if bits == 16 and not isinstance(val, int):
        val = numpy.divide(val, 257.).astype(int)
    elif bits > 16:
        val = val * 0
    if fmt.startswith('bayer_rggb'):
        expect = [
            [[64 - (bits == 16), 0, 0], [32 - (bits == 16), 0, 0]],
            [[32 - (bits == 16), 0, 0], [16 - (bits == 16), 0, 0]],
        ]
    elif fmt.startswith('bayer_bggr'):
        expect = [
            [[0, 0, 64 - (bits == 16)], [0, 0, 32 - (bits == 16)]],
            [[0, 0, 32 - (bits == 16)], [0, 0, 16 - (bits == 16)]],
        ]
    elif fmt.startswith('bayer_g'):
        expect = [
            [[0, 64 - (bits == 16), 0], [0, 16 - (bits == 16), 0]],
            [[0, 16 - (bits == 16), 0], [0, 0, 0]],
        ]
    elif fmt == 'yuv422':
        expect = [[[0, 102, 0], [0, 154, 0]], [[0, 154, 0], [0, 154, 0]]]
    else:
        assert isinstance(val, numpy.ndarray)
        zero = numpy.multiply(val, 0).tolist()
        expect = [[val.tolist(), zero], [zero, zero]]

    numpy.testing.assert_array_equal(img[11:13, 15:17], expect)


@pytest.mark.parametrize(('fmt', 'endian'), product(FORMATS, (False, True)))
def test_convert_bgr16(fmt: str, endian: bool) -> None:
    """Test with color format conversion to bgr8."""
    image, bits, val = generate_image(fmt, endian)

    # unsupported auto conversions
    if fmt[-2:] in ('C1', 'C2', 'C4') or fmt[-1] in 'USF':
        with pytest.raises(ImageConversionError):
            image_to_cvimage(image, 'bgr16')
        return

    img = image_to_cvimage(image, 'bgr16')
    assert img.shape == (image.height, image.width, 3)

    # flip rgb, remove alpha, expand mono
    if fmt.startswith('rgb'):
        val = numpy.flip(val[:3])  # type: ignore
    elif fmt.startswith('bgr'):
        assert isinstance(val, numpy.ndarray)
        val = val[:3]
    elif fmt.startswith('mono'):
        assert isinstance(val, int)
        val = numpy.array([val, val, val])

    if bits == 8 and not isinstance(val, int):
        val = numpy.multiply(val, 257.).astype(int)
    elif bits > 16:
        val = val * 0
    if fmt.startswith('bayer_rggb'):
        expect = [
            [[16448 - (bits == 16) * 64, 0, 0], [8224 - (bits == 16) * 32, 0, 0]],
            [[8224 - (bits == 16) * 32, 0, 0], [4112 - (bits == 16) * 16, 0, 0]],
        ]
    elif fmt.startswith('bayer_bggr'):
        expect = [
            [[0, 0, 16448 - (bits == 16) * 64], [0, 0, 8224 - (bits == 16) * 32]],
            [[0, 0, 8224 - (bits == 16) * 32], [0, 0, 4112 - (bits == 16) * 16]],
        ]
    elif fmt.startswith('bayer_g'):
        expect = [
            [[0, 16448 - (bits == 16) * 64, 0], [0, 4112 - (bits == 16) * 16, 0]],
            [[0, 4112 - (bits == 16) * 16, 0], [0, 0, 0]],
        ]
    elif fmt == 'yuv422':
        expect = [[[0, 26214, 0], [0, 39578, 0]], [[0, 39578, 0], [0, 39578, 0]]]
    else:
        assert isinstance(val, numpy.ndarray)
        zero = numpy.multiply(val, 0).tolist()
        expect = [[val.tolist(), zero], [zero, zero]]

    numpy.testing.assert_array_equal(img[11:13, 15:17], expect)


def test_compressed_png() -> None:
    """Test compressed png image."""
    image = CompressedImage(
        HEADER,
        'png',
        numpy.frombuffer(
            b64decode(
                """
                iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9
                awAAAABJRU5ErkJggg==
                """,
            ),
            dtype=numpy.uint8,
        ),
    )

    img = compressed_image_to_cvimage(image)
    numpy.testing.assert_array_equal(img, numpy.array([[[0, 255, 0]]]))

    img = compressed_image_to_cvimage(image, 'bgr8')
    numpy.testing.assert_array_equal(img, numpy.array([[[0, 255, 0]]]))

    img = compressed_image_to_cvimage(image, 'mono8')
    numpy.testing.assert_array_equal(img, numpy.array([[150]]))


def test_compressed_jpg() -> None:
    """Test compressed jpg image."""
    image = CompressedImage(
        HEADER,
        'jpeg',
        numpy.frombuffer(
            b64decode(
                """
                /9j/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8Q
                EBEQCgwSExIQEw8QEBD/yQALCAABAAEBAREA/8wABgAQEAX/2gAIAQEAAD8A0s8g/9k=
                """,
            ),
            dtype=numpy.uint8,
        ),
    )
    img = compressed_image_to_cvimage(image, 'bgr8')
    numpy.testing.assert_array_equal(img, numpy.array([[190]]))


def test_type_detection() -> None:
    """Test message type detection."""
    cimage = CompressedImage(
        HEADER,
        'jpeg',
        numpy.array([]),
    )
    with patch('rosbags.image.image.compressed_image_to_cvimage') as mock:
        message_to_cvimage(cimage, 'bgr8')
        assert mock.called_with(cimage, 'bgr8')

    image = Image(
        header=HEADER,
        height=-1,
        width=-1,
        encoding='',
        is_bigendian=False,
        step=-1,
        data=numpy.array([]),
    )
    with patch('rosbags.image.image.image_to_cvimage') as mock:
        message_to_cvimage(image, 'bgr8')
        assert mock.called_with(image, 'bgr8')
