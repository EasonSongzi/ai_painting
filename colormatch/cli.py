"""Command line interface.

    python -m colormatch match -r ref.jpg -t photo.jpg -o out.jpg
    python -m colormatch match -r ref.jpg -t a.jpg b.jpg -o out_dir/ --tone 0.7
    python -m colormatch match -r ref.jpg -t photo.jpg -o out.jpg --save-cube look.cube
    python -m colormatch apply -l look.cube -t a.jpg b.jpg -o out_dir/

Output keeps the input's format unless -o's extension or --format says
otherwise. TIFF is written 16-bit uncompressed, PNG lossless, JPEG at
quality 100 without chroma subsampling; ICC profile and EXIF are carried over.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .color import FORMATS, format_of, load_image, read_image, save_image
from .lut import load_cube, save_cube
from .pipeline import METHODS, apply_color_match, fit_color_match


def _out_path(out: str, target: str, many: bool, fmt: str) -> str:
    if many or out.endswith(os.sep) or os.path.isdir(out):
        os.makedirs(out, exist_ok=True)
        stem = os.path.splitext(os.path.basename(target))[0]
        return os.path.join(out, f"{stem}_matched.{fmt}")
    if format_of(out) != fmt:
        out = f"{os.path.splitext(out)[0]}.{fmt}"
    return out


def _save(args, src_path: str, rgb, meta) -> str:
    many = len(args.target) > 1
    fmt = args.format or (None if many else format_of(args.output)) or meta.format
    dst = _out_path(args.output, src_path, many, fmt)
    save_image(dst, rgb, args.quality, meta, args.bit_depth)
    return dst


def cmd_match(args) -> None:
    if args.save_cube and len(args.target) > 1:
        sys.exit("--save-cube needs a single --target (each target gets its own fit)")
    ref = load_image(args.reference)
    for path in args.target:
        t0 = time.time()
        img, meta = read_image(path)
        cm = fit_color_match(ref, img, method=args.method, lut_size=args.lut_size, seed=args.seed)
        lut = cm.lut(args.amount, args.tone, args.color)
        dst = _save(args, path, apply_color_match(img, lut, args.skin_protect), meta)
        if args.save_cube:
            save_cube(args.save_cube, lut)
        print(f"{path} -> {dst}  ({time.time() - t0:.1f}s)")


def cmd_apply(args) -> None:
    lut = load_cube(args.lut)
    for path in args.target:
        img, meta = read_image(path)
        dst = _save(args, path, apply_color_match(img, lut, args.skin_protect), meta)
        print(f"{path} -> {dst}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="colormatch", description="AI 追色: transfer the color look of photo A onto photo B")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("match", help="fit the look of a reference photo and apply it")
    m.add_argument("-r", "--reference", required=True, help="reference photo A")
    m.add_argument("-t", "--target", required=True, nargs="+", help="photo(s) B to recolor")
    m.add_argument("-o", "--output", required=True, help="output file, or directory for several targets")
    m.add_argument("--method", choices=METHODS, default="ot",
                   help="ot = covariance + sliced optimal transport (default); mkl = covariance only; reinhard = mean/std")
    m.add_argument("--amount", type=float, default=1.0, help="overall strength 0..1")
    m.add_argument("--tone", type=float, default=1.0, help="lightness/contrast strength 0..1")
    m.add_argument("--color", type=float, default=1.0, help="hue/saturation strength 0..1")
    m.add_argument("--lut-size", type=int, default=33)
    m.add_argument("--save-cube", help="also save the fitted look as a .cube 3D LUT")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_match)

    a = sub.add_parser("apply", help="apply a saved .cube look")
    a.add_argument("-l", "--lut", required=True)
    a.add_argument("-t", "--target", required=True, nargs="+")
    a.add_argument("-o", "--output", required=True)
    a.set_defaults(func=cmd_apply)

    for sp in (m, a):
        sp.add_argument("--skin-protect", type=float, default=0.0, help="0..1, keep skin tones closer to the original")
        sp.add_argument("--format", choices=FORMATS,
                        help="output format (default: from -o extension, else same as input); tif/png are lossless")
        sp.add_argument("--bit-depth", type=int, choices=(8, 16),
                        help="TIFF bit depth (default 16)")
        sp.add_argument("--quality", type=int, default=100, help="JPEG quality (default 100)")

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
