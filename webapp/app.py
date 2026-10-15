"""Local web UI for AI 追色.

    python webapp/app.py            # then open http://127.0.0.1:5050

Upload reference A and photo B once (the transform is fitted once);
slider changes only rebuild the LUT and re-render a preview, so they are
cheap.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import uuid
from collections import OrderedDict

from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colormatch.color import FORMATS, encode_image, read_image, thumbnail, to_uint8  # noqa: E402
from colormatch.lut import save_cube  # noqa: E402
from colormatch.pipeline import METHODS, apply_color_match, fit_color_match  # noqa: E402

PREVIEW_SIDE = 1600
MAX_SESSIONS = 8

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
_sessions: OrderedDict[str, dict] = OrderedDict()


def _read_upload(field: str):
    f = request.files.get(field)
    if f is None or not f.filename:
        abort(400, f"missing file '{field}'")
    try:
        return read_image(f.read(), f.filename)
    except Exception as e:
        abort(400, f"cannot read '{f.filename}': {e}")


def _params(src) -> dict:
    def num(name, default):
        try:
            return min(1.0, max(0.0, float(src.get(name, default))))
        except (TypeError, ValueError):
            return default

    return {
        "amount": num("amount", 1.0),
        "tone": num("tone", 1.0),
        "color": num("color", 1.0),
        "skin": num("skin", 0.0),
    }


def _session(token: str) -> dict:
    s = _sessions.get(token)
    if s is None:
        abort(404, "session expired, please upload again")
    _sessions.move_to_end(token)
    return s


def _render(s: dict, p: dict, full: bool) -> np.ndarray:
    lut = s["match"].lut(p["amount"], p["tone"], p["color"])
    img = s["target"] if full else s["preview"]
    return apply_color_match(img, lut, p["skin"])


def _jpeg(rgb: np.ndarray):
    """Screen preview only; downloads go through encode_image at full quality."""
    buf = io.BytesIO()
    Image.fromarray(to_uint8(rgb)).save(buf, "JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.get("/")
def index():
    return render_template("index.html", methods=METHODS)


@app.post("/api/match")
def match():
    ref, _ = _read_upload("reference")
    tgt, meta = _read_upload("target")
    method = request.form.get("method", "ot")
    if method not in METHODS:
        abort(400, "unknown method")
    token = uuid.uuid4().hex
    _sessions[token] = {
        "match": fit_color_match(ref, tgt, method=method),
        "target": tgt,
        "meta": meta,
        "preview": thumbnail(tgt, PREVIEW_SIDE),
        "name": os.path.splitext(request.files["target"].filename)[0],
    }
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    return jsonify({"token": token})


@app.get("/api/preview/<token>")
def preview(token):
    s = _session(token)
    return _jpeg(_render(s, _params(request.args), full=False))


@app.get("/api/original/<token>")
def original(token):
    return _jpeg(_session(token)["preview"])


MIMETYPES = {"jpg": "image/jpeg", "png": "image/png", "tif": "image/tiff"}


@app.get("/api/download/<token>")
def download(token):
    """Full-resolution result; ?format=jpg|png|tif (tif = 16-bit lossless)."""
    s = _session(token)
    fmt = request.args.get("format", "jpg")
    if fmt not in FORMATS:
        abort(400, "unknown format")
    data = encode_image(_render(s, _params(request.args), full=True), fmt, s["meta"])
    return send_file(io.BytesIO(data), mimetype=MIMETYPES[fmt], as_attachment=True,
                     download_name=f"{s['name']}_matched.{fmt}")


@app.get("/api/cube/<token>")
def cube(token):
    s = _session(token)
    p = _params(request.args)
    with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as tmp:
        path = tmp.name
    try:
        save_cube(path, s["match"].lut(p["amount"], p["tone"], p["color"]))
        data = io.BytesIO(open(path, "rb").read())
    finally:
        os.unlink(path)
    return send_file(data, mimetype="text/plain", as_attachment=True, download_name=f"{s['name']}_look.cube")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5050)), debug=False)
