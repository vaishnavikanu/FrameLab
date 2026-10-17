"""Per-frame content complexity: spatial detail + motion-compensated temporal residual.

Spatial   = mean absolute gradient of the (downscaled) luma frame        [native: gradient_energy]
Temporal  = mean per-pixel SAD left after block motion compensation       [native: motion_estimate]
Frames whose compensated residual spikes far above the video's median are flagged as scene cuts.
These are cheap proxies for "how hard is this content to encode".
"""
import time

import cv2
import numpy as np
import pandas as pd

from . import kernels
from .video_io import open_video


def _prep(frame, width):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if width and g.shape[1] > width:
        h = max(2, int(round(g.shape[0] * width / g.shape[1])))
        g = cv2.resize(g, (int(width), h), interpolation=cv2.INTER_AREA)
    return g


def analyze_complexity(path, max_frames=None, analysis_width=640, block=16, search=16,
                       method="diamond", cut_ratio=4.0, cut_min=5.0, prefer_native=True):
    cap, meta = open_video(path)
    rows, prev, idx = [], None, 0
    t_start = time.perf_counter()
    try:
        while max_frames is None or idx < int(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            gray = _prep(frame, analysis_width)
            row = {"frame_index": idx,
                   "spatial": kernels.gradient_energy(gray, prefer_native=prefer_native),
                   "temporal_residual": np.nan, "zero_motion_diff": np.nan, "mean_motion": np.nan}
            if prev is not None:
                mvs, sads = kernels.motion_estimate(prev, gray, block, search, method,
                                                    prefer_native=prefer_native)
                row["temporal_residual"] = float(sads.mean()) / (block * block)
                row["zero_motion_diff"] = float(cv2.absdiff(prev, gray).mean())
                row["mean_motion"] = float(np.hypot(mvs[..., 0], mvs[..., 1]).mean())
            rows.append(row)
            prev, idx = gray, idx + 1
    finally:
        cap.release()
    if not rows:
        raise ValueError("No frames were decoded from the input video")
    df = pd.DataFrame(rows)
    # Adaptive cut detector: a frame is a cut if its motion-compensated residual is both
    # clearly above the noise floor (cut_min) and a spike relative to this video's typical residual.
    median = float(df["temporal_residual"].median()) if len(df) > 1 else 0.0
    cut_threshold = max(float(cut_min), float(cut_ratio) * median)
    df["is_scene_cut"] = df["temporal_residual"] > cut_threshold
    spatial = float(df["spatial"].mean())
    temporal = float(df["temporal_residual"].mean()) if len(df) > 1 else 0.0
    summary = {
        "frames_analyzed": len(df), "analysis_width": analysis_width,
        "native_kernels": bool(prefer_native and kernels.native_available()),
        "mean_spatial_complexity": spatial, "mean_temporal_residual": temporal,
        "mean_motion_pixels": float(df["mean_motion"].mean()) if len(df) > 1 else 0.0,
        "scene_cuts": int(df["is_scene_cut"].sum()), "scene_cut_threshold": cut_threshold,
        "scene_cut_frames": [int(i) for i in df.loc[df["is_scene_cut"], "frame_index"]],
        # Simple explainable index (8-bit intensity units); NOT a calibrated predictor.
        "complexity_index": spatial + temporal,
        "analysis_seconds": time.perf_counter() - t_start, "metadata": meta,
    }
    return summary, df
