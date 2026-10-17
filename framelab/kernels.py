"""Compute kernels with a native (C++) fast path and a pure-Python reference for each.

Every native function has a `*_python` twin that produces the *same* output, so the tests can
check them against each other and the benchmarks compare like with like.
"""
import cv2
import numpy as np

try:  # compiled by `python setup.py build_ext --inplace` or CMake
    from . import _cpp
except ImportError:
    _cpp = None


def native_available():
    return _cpp is not None


def _u8(a):
    return np.ascontiguousarray(a, dtype=np.uint8)


# ------------------------------------------------------------------ brightness / contrast
def adjust_brightness_contrast_numpy(image, alpha=1.0, beta=0.0):
    """Straightforward float32 baseline: saturate(rint(image*alpha + beta))."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    out = np.rint(arr.astype(np.float32) * np.float32(alpha) + np.float32(beta))
    return np.clip(out, 0, 255).astype(np.uint8)


def build_lut(alpha, beta):
    """256-entry lookup table, computed with the same float32 arithmetic as the baseline."""
    x = np.arange(256, dtype=np.float32)
    return np.clip(np.rint(x * np.float32(alpha) + np.float32(beta)), 0, 255).astype(np.uint8)


def adjust_brightness_contrast_lut(image, alpha=1.0, beta=0.0):
    """LUT version using OpenCV's optimised table lookup (pure Python call, no native module)."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    return cv2.LUT(arr, build_lut(alpha, beta))


def adjust_brightness_contrast(image, alpha=1.0, beta=0.0, prefer_native=True):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    if prefer_native and _cpp is not None:
        return _cpp.adjust_brightness_contrast(arr, float(alpha), float(beta))
    return adjust_brightness_contrast_lut(arr, alpha, beta)


# ------------------------------------------------------------------ motion estimation
_LDSP = ((0, -2), (-1, -1), (1, -1), (-2, 0), (2, 0), (-1, 1), (1, 1), (0, 2))
_SDSP = ((0, -1), (-1, 0), (1, 0), (0, 1))


def motion_estimate_python(prev, cur, block=16, search=16, method="diamond"):
    """Reference block matcher. Same algorithm and tie-breaking as the C++ version, just slow."""
    if prev.shape != cur.shape or prev.ndim != 2:
        raise ValueError("prev and cur must be 2D arrays of the same shape")
    if method not in ("diamond", "full"):
        raise ValueError("method must be 'diamond' or 'full'")
    if block < 2 or search < 1:
        raise ValueError("block must be >= 2 and search >= 1")
    H, W = cur.shape
    if H < block or W < block:
        raise ValueError("frame smaller than one block")
    p = prev.astype(np.int16)
    c = cur.astype(np.int16)
    nby, nbx = H // block, W // block
    mvs = np.zeros((nby, nbx, 2), np.int32)
    sads = np.zeros((nby, nbx), np.int32)

    def ok(bx, by, dx, dy):
        x, y = bx + dx, by + dy
        return abs(dx) <= search and abs(dy) <= search and 0 <= x <= W - block and 0 <= y <= H - block

    for by_i in range(nby):
        for bx_i in range(nbx):
            bx, by = bx_i * block, by_i * block
            cb = c[by:by + block, bx:bx + block]

            def sad(dx, dy):
                rb = p[by + dy:by + dy + block, bx + dx:bx + dx + block]
                return int(np.abs(cb - rb).sum())

            best = sad(0, 0)
            bdx = bdy = 0
            if method == "full":
                for dy in range(-search, search + 1):
                    for dx in range(-search, search + 1):
                        if (dx or dy) and ok(bx, by, dx, dy):
                            s = sad(dx, dy)
                            if s < best:
                                best, bdx, bdy = s, dx, dy
            else:
                cx = cy = 0
                while True:
                    ndx, ndy = cx, cy
                    for px, py_ in _LDSP:
                        dx, dy = cx + px, cy + py_
                        if ok(bx, by, dx, dy):
                            s = sad(dx, dy)
                            if s < best:
                                best, ndx, ndy = s, dx, dy
                    if (ndx, ndy) == (cx, cy):
                        break
                    cx, cy = ndx, ndy
                bdx, bdy = cx, cy
                for px, py_ in _SDSP:
                    dx, dy = cx + px, cy + py_
                    if ok(bx, by, dx, dy):
                        s = sad(dx, dy)
                        if s < best:
                            best, bdx, bdy = s, dx, dy
            mvs[by_i, bx_i] = (bdx, bdy)
            sads[by_i, bx_i] = best
    return mvs, sads


def motion_estimate(prev, cur, block=16, search=16, method="diamond", prefer_native=True):
    """For each block of `cur`, find where it came from in `prev`.

    Returns (mvs[nby, nbx, 2] as (dx, dy), sads[nby, nbx]).
    """
    prev, cur = _u8(prev), _u8(cur)
    if prefer_native and _cpp is not None:
        return _cpp.motion_estimate(prev, cur, int(block), int(search), method)
    return motion_estimate_python(prev, cur, block, search, method)


# ------------------------------------------------------------------ spatial complexity
def gradient_energy_python(gray):
    g = np.asarray(gray).astype(np.int32)
    if g.ndim != 2 or g.shape[0] < 2 or g.shape[1] < 2:
        raise ValueError("gray must be a 2D image of at least 2x2")
    dx = np.abs(np.diff(g, axis=1))[:-1, :]
    dy = np.abs(np.diff(g, axis=0))[:, :-1]
    return float((dx.sum(dtype=np.int64) + dy.sum(dtype=np.int64)) / dx.size)


def gradient_energy(gray, prefer_native=True):
    gray = _u8(gray)
    if prefer_native and _cpp is not None:
        return float(_cpp.gradient_energy(gray))
    return gradient_energy_python(gray)
