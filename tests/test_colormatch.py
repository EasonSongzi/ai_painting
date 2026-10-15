import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colormatch.cli import main as cli_main  # noqa: E402
from colormatch.color import ImageMeta, read_image, rgb_to_lab, save_image  # noqa: E402
from colormatch.lut import apply_lut, identity_lut, load_cube, save_cube  # noqa: E402
from colormatch.pipeline import METHODS, apply_color_match, color_match, fit_color_match  # noqa: E402


def _photo(seed, tint):
    """Smooth synthetic 'photo' with a gradient, some objects and a color cast."""
    rng = np.random.default_rng(seed)
    h, w = 120, 160
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    img = np.stack([x, y, 0.5 * (x + y)], -1)
    for _ in range(6):
        cy, cx, r = rng.uniform(0, 1, 3) * [1, 1, 0.3]
        img[(y - cy) ** 2 + (x - cx) ** 2 < r**2] = rng.uniform(0, 1, 3)
    return np.clip(img * 0.8 + np.asarray(tint), 0, 1).astype(np.float32)


@pytest.fixture(scope="module")
def pair():
    return _photo(1, [0.15, 0.05, -0.1]), _photo(2, [-0.1, 0.0, 0.15])


def test_identity_lut_is_exact():
    img = np.random.default_rng(0).uniform(0, 1, (20, 30, 3)).astype(np.float32)
    assert np.abs(apply_lut(img, identity_lut(17)) - img).max() < 1e-5


@pytest.mark.parametrize("method", METHODS)
def test_moves_statistics_toward_reference(pair, method):
    ref, tgt = pair
    out, _ = color_match(ref, tgt, method=method)
    ref_mu, tgt_mu, out_mu = (rgb_to_lab(i).reshape(-1, 3).mean(0) for i in (ref, tgt, out))
    assert np.linalg.norm(out_mu - ref_mu) < 0.35 * np.linalg.norm(tgt_mu - ref_mu)


def test_self_match_is_near_identity(pair):
    ref, _ = pair
    out, _ = color_match(ref, ref)
    assert np.abs(out - ref).mean() < 0.02


def test_zero_strength_is_identity(pair):
    ref, tgt = pair
    lut = fit_color_match(ref, tgt).lut(amount=0.0)
    assert np.abs(apply_color_match(tgt, lut) - tgt).max() < 0.01


def test_tone_only_keeps_chroma(pair):
    ref, tgt = pair
    lut = fit_color_match(ref, tgt).lut(tone=1.0, color=0.0)
    lab_in, lab_out = rgb_to_lab(tgt), rgb_to_lab(apply_color_match(tgt, lut))
    assert np.abs(lab_out[..., 1:] - lab_in[..., 1:]).mean() < 1.5
    assert np.abs(lab_out[..., 0] - lab_in[..., 0]).mean() > 1.0


def test_cube_round_trip(tmp_path, pair):
    ref, tgt = pair
    lut = fit_color_match(ref, tgt, lut_size=9).lut()
    path = tmp_path / "look.cube"
    save_cube(str(path), lut)
    assert np.abs(load_cube(str(path)) - lut).max() < 1e-5


def test_tiff_keeps_16_bit_precision(tmp_path):
    rgb = np.random.default_rng(0).uniform(0, 1, (30, 40, 3)).astype(np.float32)
    path = str(tmp_path / "x.tif")
    save_image(path, rgb, meta=ImageMeta(icc=b"fake-icc"))
    back, meta = read_image(path)
    assert meta.bit_depth == 16 and meta.icc == b"fake-icc"
    assert np.abs(back - rgb).max() < 1.0 / 65535 + 1e-6  # 8-bit would be ~2e-3


def test_png_is_lossless_and_keeps_icc_exif(tmp_path):
    from PIL import Image

    rgb = (np.random.default_rng(1).integers(0, 256, (30, 40, 3)) / 255.0).astype(np.float32)
    exif = Image.Exif()
    exif[0x010F] = "TestCam"  # Make
    path = str(tmp_path / "x.png")
    save_image(path, rgb, meta=ImageMeta(icc=b"fake-icc", exif=exif.tobytes()))
    back, meta = read_image(path)
    assert np.array_equal(back, rgb)
    assert meta.icc == b"fake-icc"
    assert Image.open(path).getexif()[0x010F] == "TestCam"


def test_cli_16_bit_tiff_in_tiff_out(tmp_path, pair):
    ref, tgt = pair
    save_image(str(tmp_path / "ref.png"), ref)
    save_image(str(tmp_path / "b.tif"), tgt)
    cli_main(["match", "-r", str(tmp_path / "ref.png"), "-t", str(tmp_path / "b.tif"), "-o", str(tmp_path / "out") + os.sep])
    out, meta = read_image(str(tmp_path / "out" / "b_matched.tif"))
    assert meta.bit_depth == 16 and out.shape == tgt.shape
    # A float result saved at 16-bit carries far more than 256 levels per channel.
    assert len(np.unique(out[..., 0])) > 256
