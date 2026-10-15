"""Color-space helpers (sRGB <-> CIELAB) and image I/O."""
from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass

import numpy as np
import tifffile
from PIL import Image, ImageOps
from skimage import color as skcolor

FORMATS = ("jpg", "png", "tif")
_EXT = {".jpg": "jpg", ".jpeg": "jpg", ".png": "png", ".tif": "tif", ".tiff": "tif"}
_TAG_ORIENTATION, _TAG_ICC = 274, 34675


@dataclass
class ImageMeta:
    """What must survive a round trip so the output is not degraded."""

    format: str = "jpg"
    bit_depth: int = 8
    icc: bytes | None = None
    exif: bytes | None = None


def format_of(name: str) -> str | None:
    return _EXT.get(os.path.splitext(name)[1].lower())


def _to_float_rgb(arr: np.ndarray) -> tuple[np.ndarray, int]:
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)  # planar TIFF
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    arr = arr[..., :3]
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0, 16
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0, 8
    return np.clip(arr.astype(np.float32), 0.0, 1.0), 16


def _orient(arr: np.ndarray, o: int) -> np.ndarray:
    """Apply an EXIF/TIFF orientation value (1-8) to an H x W x C array."""
    if o in (2, 4, 5, 7):
        arr = arr[:, ::-1]
    return np.rot90(arr, {3: 2, 4: 2, 5: 1, 6: 3, 7: 3, 8: 1}.get(o, 0)).copy()


def _read_tiff(src) -> tuple[np.ndarray, ImageMeta]:
    with tifffile.TiffFile(src) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        # Tag values load lazily, so read them before the file closes.
        icc = page.tags.get(_TAG_ICC)
        icc = bytes(icc.value) if icc is not None else None
        orient = page.tags.get(_TAG_ORIENTATION)
        orient = int(orient.value) if orient is not None else 1
    rgb, depth = _to_float_rgb(arr)
    return _orient(rgb, orient), ImageMeta("tif", depth, icc)


def read_image(source, name: str | None = None) -> tuple[np.ndarray, ImageMeta]:
    """Load float32 RGB in [0, 1] plus metadata, keeping 16-bit precision.

    `source` is a path, bytes or a binary file object; `name` supplies the
    extension when it is not a path.
    """
    name = name or (source if isinstance(source, str) else "")
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    fmt = format_of(name) or "jpg"
    if fmt == "tif":
        try:
            return _read_tiff(source)
        except Exception:
            if hasattr(source, "seek"):
                source.seek(0)  # e.g. JPEG-compressed TIFF: let Pillow try
    img = ImageOps.exif_transpose(Image.open(source))
    exif = img.getexif()
    exif.pop(_TAG_ORIENTATION, None)  # pixels are already upright
    meta = ImageMeta(fmt, 8, img.info.get("icc_profile"), exif.tobytes() if len(exif) else None)
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        rgb, meta.bit_depth = _to_float_rgb(np.asarray(img).astype(np.uint16))
        return rgb, meta
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0, meta


def load_image(path: str, max_side: int | None = None) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1], honoring orientation."""
    rgb, _ = read_image(path)
    return thumbnail(rgb, max_side) if max_side else rgb


def encode_image(
    rgb: np.ndarray,
    fmt: str,
    meta: ImageMeta | None = None,
    quality: int = 100,
    bit_depth: int | None = None,
) -> bytes:
    """Encode losslessly where the format allows it.

    tif: 16-bit (default) or 8-bit, uncompressed; png: 8-bit lossless;
    jpg: no chroma subsampling. ICC profile and EXIF are carried over.
    """
    meta = meta or ImageMeta()
    buf = io.BytesIO()
    if fmt == "tif":
        depth = bit_depth or 16
        arr = to_uint16(rgb) if depth == 16 else to_uint8(rgb)
        extratags = [(_TAG_ICC, "B", len(meta.icc), meta.icc, True)] if meta.icc else []
        tifffile.imwrite(buf, arr, photometric="rgb", extratags=extratags)
        return buf.getvalue()
    if bit_depth == 16:
        raise ValueError("16-bit output needs TIFF (.tif)")
    img = Image.fromarray(to_uint8(rgb))
    kw = {k: v for k, v in (("icc_profile", meta.icc), ("exif", meta.exif)) if v}
    if fmt == "jpg":
        img.save(buf, "JPEG", quality=quality, subsampling=0, **kw)
    elif fmt == "png":
        img.save(buf, "PNG", **kw)
    else:
        raise ValueError(f"unknown format {fmt!r}; use one of {FORMATS}")
    return buf.getvalue()


def save_image(
    path: str,
    rgb: np.ndarray,
    quality: int = 100,
    meta: ImageMeta | None = None,
    bit_depth: int | None = None,
) -> None:
    fmt = format_of(path)
    if fmt is None:
        raise ValueError(f"{path}: use a .jpg, .png or .tif extension")
    data = encode_image(rgb, fmt, meta, quality, bit_depth)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def thumbnail(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    img = Image.fromarray(to_uint8(rgb))
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return skcolor.rgb2lab(np.clip(rgb, 0.0, 1.0)).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    # Transferred colors can fall outside the sRGB gamut; clipping is intended.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rgb = skcolor.lab2rgb(lab)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# Lab channels have very different ranges; scale them so distribution
# matching treats lightness and chroma comparably.
LAB_SCALE = np.array([1.0 / 100.0, 1.0 / 128.0, 1.0 / 128.0], dtype=np.float32)


def rgb_to_nlab(rgb: np.ndarray) -> np.ndarray:
    return rgb_to_lab(rgb) * LAB_SCALE


def nlab_to_rgb(nlab: np.ndarray) -> np.ndarray:
    return lab_to_rgb(nlab / LAB_SCALE)
