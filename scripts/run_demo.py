"""One-command demo: generates a sample video, then runs every FrameLab feature on it.

    python scripts/run_demo.py

Everything is written to outputs/demo/. Takes about a minute.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framelab import kernels  # noqa: E402
from framelab.benchmarks import kernel_benchmark, motion_benchmark  # noqa: E402
from framelab.complexity import analyze_complexity  # noqa: E402
from framelab.ffmpeg_tools import capabilities, have_x264  # noqa: E402
from framelab.pipeline import process_video  # noqa: E402
from framelab.sweep import crf_sweep  # noqa: E402

OUT = ROOT / "outputs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
sample = ROOT / "samples" / "sample.mp4"
if not sample.exists():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_video.py")], check=True)

print("== system:", json.dumps({"native_cpp": kernels.native_available(), **capabilities()}, default=str))

print("\n== complexity analysis")
s, df = analyze_complexity(sample)
df.to_csv(OUT / "complexity.csv", index=False)
print(f"complexity index {s['complexity_index']:.1f} | scene cuts at frames {s['scene_cut_frames']} | "
      f"{s['analysis_seconds']:.2f}s ({'C++' if s['native_kernels'] else 'Python'} kernels)")

print("\n== enhancement methods (before -> after)")
print(f"{'method':<11}{'enh ms/frame':>13}{'blur':>16}{'noise':>14}{'PSNR vs in':>12}")
for m in ("identity", "clahe", "sharpen", "brightness", "denoise"):
    r = process_video(sample, OUT / f"sample_{m}.mp4", method=m, max_frames=40)
    print(f"{m:<11}{r['mean_enhance_ms']:>13.1f}"
          f"{r['mean_blur_laplacian_variance_in']:>8.0f}->{r['mean_blur_laplacian_variance_out']:<6.0f}"
          f"{r['mean_noise_residual_std_in']:>7.2f}->{r['mean_noise_residual_std_out']:<5.2f}"
          f"{r['mean_psnr_vs_input_db']:>9.1f} dB")

if have_x264():
    print("\n== content-aware encoding sweep")
    r = crf_sweep(sample, OUT / "sweep", max_frames=150)
    print(r["note"], f"({r['metric']}, saves {r.get('bitrate_saving_vs_lowest_crf_pct', 0):.0f}% bitrate vs CRF {r['table'][0]['crf']})")
    print("curve:", r["plot"])
else:
    print("\n(skipping CRF sweep: ffmpeg with libx264 not found; `pip install imageio-ffmpeg`)")

print("\n== kernel benchmarks")
print(json.dumps(kernel_benchmark(1920, 1080, 20)["ms_per_call"], indent=2))
mb = motion_benchmark(640, 360, 3)
print(json.dumps(mb["ms_per_frame_pair"], indent=2))
print(f"\nDone. Open the dashboard with:  streamlit run app/dashboard.py   (outputs in {OUT})")
