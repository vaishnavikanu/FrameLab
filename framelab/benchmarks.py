"""Honest micro-benchmarks. Timings depend on hardware; correctness flags are what matter most."""
import time

import cv2
import numpy as np

from . import kernels


def _time(fn, repeats):
    fn()  # warm-up
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
    return (time.perf_counter() - t0) / repeats * 1000.0, out


def kernel_benchmark(width=1920, height=1080, repeats=20, alpha=1.1, beta=8.0):
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    impls = {"numpy_float32": lambda: kernels.adjust_brightness_contrast_numpy(image, alpha, beta),
             "opencv_lut": lambda: kernels.adjust_brightness_contrast_lut(image, alpha, beta)}
    if kernels.native_available():
        impls["cpp_lut"] = lambda: kernels.adjust_brightness_contrast(image, alpha, beta, prefer_native=True)
    ms, outs = {}, {}
    for name, fn in impls.items():
        ms[name], outs[name] = _time(fn, repeats)
    base = ms["numpy_float32"]
    return {"image_shape": [height, width, 3], "repeats": repeats, "native_available": kernels.native_available(),
            "ms_per_call": ms,
            "speedup_vs_numpy": {k: base / v for k, v in ms.items()},
            "identical_to_numpy": {k: bool(np.array_equal(outs["numpy_float32"], v)) for k, v in outs.items()}}


def _textured_pair(width, height, shift=(3, -2), seed=0):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (0, 0), 2.5)
    return base, np.roll(base, (shift[1], shift[0]), (0, 1))  # (rows, cols)


def motion_benchmark(width=640, height=360, repeats=3, block=16, search=16, include_python_full=False):
    prev, cur = _textured_pair(width, height)
    runs = {"python_diamond": lambda: kernels.motion_estimate(prev, cur, block, search, "diamond", prefer_native=False)}
    if kernels.native_available():
        runs["cpp_diamond"] = lambda: kernels.motion_estimate(prev, cur, block, search, "diamond")
        runs["cpp_full"] = lambda: kernels.motion_estimate(prev, cur, block, search, "full")
    if include_python_full:
        runs["python_full"] = lambda: kernels.motion_estimate(prev, cur, block, search, "full", prefer_native=False)
    ms, outs = {}, {}
    for name, fn in runs.items():
        ms[name], outs[name] = _time(fn, 1 if name.startswith("python") else repeats)
    res = {"frame_shape": [height, width], "block": block, "search_range": search,
           "ms_per_frame_pair": ms, "mean_sad_per_block": {k: float(v[1].mean()) for k, v in outs.items()}}
    if "cpp_diamond" in ms:
        res["speedup_cpp_diamond_vs_python_diamond"] = ms["python_diamond"] / ms["cpp_diamond"]
        res["python_equals_cpp_diamond"] = bool(
            np.array_equal(outs["python_diamond"][0], outs["cpp_diamond"][0])
            and np.array_equal(outs["python_diamond"][1], outs["cpp_diamond"][1]))
        res["speedup_diamond_vs_full_search"] = ms["cpp_full"] / ms["cpp_diamond"]
        res["diamond_sad_vs_full_sad_ratio"] = float(outs["cpp_diamond"][1].mean() / max(outs["cpp_full"][1].mean(), 1e-9))
    if "python_full" in outs and "cpp_full" in outs:
        res["python_equals_cpp_full"] = bool(np.array_equal(outs["python_full"][0], outs["cpp_full"][0]))
    if not kernels.native_available():
        res["note"] = "Native module not built: only the Python reference was timed. See README step 'Build the C++ core'."
    return res
