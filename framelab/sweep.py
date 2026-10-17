"""Content-aware encoding study: sweep x264 CRF values, measure bitrate vs quality, and pick the
cheapest setting that still meets a quality target."""
import tempfile
import time
from pathlib import Path

import pandas as pd

from .complexity import analyze_complexity
from .ffmpeg_tools import compute_vmaf, encode_x264, have_vmaf, have_x264
from .pipeline import compare_videos
from .video_io import open_video

DEFAULT_TARGETS = {"vmaf": 93.0, "ssim": 0.95, "psnr": 38.0}
METRIC_COLUMN = {"vmaf": "vmaf", "ssim": "mean_ssim", "psnr": "mean_psnr_db"}


def crf_sweep(input_path, output_dir, crfs=(18, 20, 22, 24, 26, 28, 30, 32, 34), preset="medium", max_frames=150,
              metric="auto", target=None, keep_encodes=False, with_complexity=True):
    """Encode `input_path` at each CRF, score it against the source, recommend a CRF.

    metric: 'vmaf' (needs an ffmpeg with libvmaf), 'ssim', 'psnr', or 'auto' (vmaf if available).
    The recommended CRF is the highest (smallest file) whose quality is >= target.
    """
    if not have_x264():
        raise RuntimeError("CRF sweep needs ffmpeg with libx264 (pip install imageio-ffmpeg).")
    if metric == "auto":
        metric = "vmaf" if have_vmaf() else "ssim"
    if metric == "vmaf" and not have_vmaf():
        raise RuntimeError("This ffmpeg build has no libvmaf; use metric='ssim' or 'psnr'.")
    if metric not in METRIC_COLUMN:
        raise ValueError("metric must be auto, vmaf, ssim or psnr")
    target = DEFAULT_TARGETS[metric] if target is None else float(target)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, meta = open_video(input_path)
    n = meta["frame_count_reported"]
    frames = min(n, int(max_frames)) if max_frames and n > 0 else (n or None)
    if not frames:
        raise ValueError("Could not determine the number of frames to encode.")
    duration = frames / meta["fps"]

    rows = []
    with tempfile.TemporaryDirectory(prefix="framelab_sweep_") as tmp:
        for crf in sorted(int(c) for c in crfs):
            enc = Path(out_dir if keep_encodes else tmp) / f"crf{crf}.mp4"
            t0 = time.perf_counter()
            encode_x264(input_path, enc, crf=crf, preset=preset, max_frames=frames)
            enc_s = time.perf_counter() - t0
            cmp = compare_videos(input_path, enc, Path(tmp) / f"cmp{crf}", max_frames=frames)
            row = {"crf": crf, "size_kb": enc.stat().st_size / 1024.0,
                   "bitrate_kbps": enc.stat().st_size * 8 / duration / 1000.0,
                   "encode_seconds": enc_s, "mean_psnr_db": cmp["mean_psnr_db"],
                   "mean_ssim": cmp["mean_ssim"]}
            if metric == "vmaf":
                row["vmaf"] = compute_vmaf(enc, input_path, max_frames=frames, work_dir=Path(tmp) / f"v{crf}")
            rows.append(row)
    df = pd.DataFrame(rows)
    col = METRIC_COLUMN[metric]
    passing = df[df[col] >= target]
    recommended = int(passing["crf"].max()) if not passing.empty else None
    result = {"input": str(input_path), "frames_encoded": frames, "preset": preset, "metric": metric,
              "target": target, "recommended_crf": recommended,
              "note": ("No CRF met the target; try lower CRF values." if recommended is None else
                       f"CRF {recommended} is the smallest file that still meets {metric} >= {target}.")}
    if recommended is not None:
        base = df[df["crf"] == df["crf"].min()].iloc[0]
        best = df[df["crf"] == recommended].iloc[0]
        result["bitrate_saving_vs_lowest_crf_pct"] = float(100 * (1 - best["bitrate_kbps"] / base["bitrate_kbps"]))
    if with_complexity:
        summary, _ = analyze_complexity(input_path, max_frames=frames)
        result["complexity"] = {k: summary[k] for k in
                                ("mean_spatial_complexity", "mean_temporal_residual", "scene_cuts", "complexity_index")}
    csv = out_dir / "crf_sweep.csv"
    df.to_csv(csv, index=False)
    result["csv"] = str(csv)
    result["plot"] = plot_rd_curve(df, metric, target, recommended, out_dir / "rd_curve.png")
    result["table"] = df.to_dict(orient="records")
    return result


def plot_rd_curve(df, metric, target, recommended, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = METRIC_COLUMN[metric]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    ax.plot(df["bitrate_kbps"], df[col], "o-")
    for _, r in df.iterrows():
        ax.annotate(f"CRF {int(r['crf'])}", (r["bitrate_kbps"], r[col]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.axhline(target, color="tab:red", ls="--", lw=1, label=f"target {target:g}")
    ax.set_xlabel("Bitrate (kbit/s)")
    ax.set_ylabel(metric.upper())
    ax.set_title("Rate-distortion curve (x264)" + (f"  |  recommended CRF {recommended}" if recommended else ""))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)
