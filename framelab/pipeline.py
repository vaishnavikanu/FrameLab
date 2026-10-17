"""Sequential video processing, inspection and reference comparison."""
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .analysis import analyze_frame
from .enhancement import apply_enhancement
from .metrics import psnr, ssim
from .video_io import create_writer, open_video


def _progress(cb, count, total):
    if cb:
        cb(min(count / total, 1.0) if total > 0 else None, count)


def process_video(input_path, output_path, method="clahe", progress_callback=None, max_frames=None,
                  params=None, analyze=True, backend="auto", crf=18, preset="medium"):
    """Enhance a video frame by frame.

    Timing is reported separately: `mean_enhance_ms` is only the enhancement call, while
    `fps_processed` is end-to-end (decode + enhance + analysis + encode).
    With `analyze=True` each frame also gets before/after quality indicators, which is how you
    check whether an enhancement actually changed blur/noise/brightness as intended.
    """
    params = params or {}
    cap, meta = open_video(input_path)
    width, height, fps = meta["width"], meta["height"], meta["fps"]
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video reports invalid dimensions")
    writer = create_writer(output_path, fps, width, height, backend=backend, crf=crf,
                           preset=preset, audio_source=input_path if max_frames is None else None)
    rows, enhance_ms = [], []
    start = time.perf_counter()
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and count >= int(max_frames)):
                break
            t0 = time.perf_counter()
            enhanced = apply_enhancement(frame, method=method, **params)
            ms = (time.perf_counter() - t0) * 1000.0
            enhance_ms.append(ms)
            writer.write(enhanced)
            row = {"frame_index": count, "enhance_ms": ms}
            if analyze:
                a_in, a_out = analyze_frame(frame), analyze_frame(enhanced)
                row.update({f"in_{k}": v for k, v in a_in.items()})
                row.update({f"out_{k}": v for k, v in a_out.items()})
                row["psnr_vs_input_db"] = min(psnr(frame, enhanced), 100.0)
            rows.append(row)
            count += 1
            _progress(progress_callback, count, meta["frame_count_reported"])
    finally:
        cap.release()
        try:
            writer.release()
        except Exception:
            if count > 0:      # a real failure while encoding: surface it
                raise
    wall = time.perf_counter() - start
    if count == 0:
        raise ValueError("No frames were decoded from the input video")

    out = Path(output_path)
    report_path = out.with_suffix(".csv")
    df = pd.DataFrame(rows)
    df.to_csv(report_path, index=False)
    size = out.stat().st_size
    duration = count / fps
    result = {
        "input": str(input_path), "output": str(output_path), "csv": str(report_path),
        "method": method, "frames_processed": count, "writer_backend": writer.backend,
        "audio_preserved": bool(getattr(writer, "audio", False)),
        "wall_time_seconds": wall, "fps_processed": count / wall if wall > 0 else 0.0,
        "mean_enhance_ms": float(np.mean(enhance_ms)),
        "p95_enhance_ms": float(np.percentile(enhance_ms, 95)),
        "realtime_capable": bool((count / wall) >= fps),
        "output_size_bytes": size, "output_bitrate_kbps": size * 8 / duration / 1000.0,
        "metadata": meta, "frame_metrics": rows,
    }
    if analyze:
        for key in ("brightness_mean", "blur_laplacian_variance", "noise_residual_std"):
            result[f"mean_{key}_in"] = float(df[f"in_{key}"].mean())
            result[f"mean_{key}_out"] = float(df[f"out_{key}"].mean())
        result["mean_psnr_vs_input_db"] = float(df["psnr_vs_input_db"].mean())
    return result


def inspect_video(input_path, output_dir, max_frames=None, progress_callback=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap, meta = open_video(input_path)
    rows, count = [], 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and count >= int(max_frames)):
                break
            rows.append({"frame_index": count, **analyze_frame(frame)})
            count += 1
            _progress(progress_callback, count, meta["frame_count_reported"])
    finally:
        cap.release()
    if count == 0:
        raise ValueError("No frames were decoded from the input video")
    csv_path = output_dir / "inspection.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"metadata": meta, "frames_processed": count,
            "wall_time_seconds": time.perf_counter() - start,
            "csv": str(csv_path), "frame_metrics": rows}


def _align(a, b, idx):
    """Frames may differ by one pixel when an encoder cropped to even size; crop both to match."""
    if a.shape == b.shape:
        return a, b
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    if abs(a.shape[0] - b.shape[0]) > 1 or abs(a.shape[1] - b.shape[1]) > 1:
        raise ValueError(f"Frame dimensions differ at frame {idx}: {a.shape} vs {b.shape}")
    return a[:h, :w], b[:h, :w]


def compare_videos(reference_path, candidate_path, output_dir, max_frames=None):
    """PSNR/SSIM between corresponding frames of two videos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref, _ = open_video(reference_path)
    cand, _ = open_video(candidate_path)
    rows, idx, length_mismatch = [], 0, False
    try:
        while max_frames is None or idx < int(max_frames):
            ok1, a = ref.read()
            ok2, b = cand.read()
            if ok1 != ok2:
                length_mismatch = True
            if not ok1 or not ok2:
                break
            a, b = _align(a, b, idx)
            rows.append({"frame_index": idx, "psnr_db": min(psnr(a, b), 100.0), "ssim": ssim(a, b)})
            idx += 1
    finally:
        ref.release()
        cand.release()
    if not rows:
        raise ValueError("No matching frames found")
    csv_path = output_dir / "reference_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"frames_compared": len(rows), "length_mismatch": length_mismatch,
            "mean_psnr_db": float(np.mean([r["psnr_db"] for r in rows])),
            "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
            "csv": str(csv_path), "frame_metrics": rows}
