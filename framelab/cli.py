"""Command-line entry point:  python -m framelab.cli <command> ..."""
import argparse
import json
import sys

from . import __version__, kernels
from .benchmarks import kernel_benchmark, motion_benchmark
from .complexity import analyze_complexity
from .enhancement import METHODS
from .ffmpeg_tools import capabilities
from .pipeline import compare_videos, inspect_video, process_video
from .sweep import crf_sweep


def _dump(obj):
    print(json.dumps(obj, indent=2, default=str))


def build_parser():
    parser = argparse.ArgumentParser(prog="framelab", description="FrameLab: video enhancement, "
                                     "complexity analysis and content-aware encoding study")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show native-module and ffmpeg/VMAF availability")

    p = sub.add_parser("inspect", help="Per-frame brightness/blur/noise indicators -> CSV")
    p.add_argument("input")
    p.add_argument("--output", default="outputs/inspection")
    p.add_argument("--max-frames", type=int)

    p = sub.add_parser("process", help="Enhance a video (keeps audio when ffmpeg is available)")
    p.add_argument("input")
    p.add_argument("--method", choices=METHODS, default="clahe")
    p.add_argument("--output", default="outputs/enhanced.mp4")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--backend", choices=["auto", "ffmpeg", "opencv"], default="auto")
    p.add_argument("--crf", type=int, default=18, help="x264 quality for the output (lower = better)")
    p.add_argument("--no-analyze", action="store_true", help="skip per-frame before/after metrics (faster)")

    p = sub.add_parser("compare", help="PSNR/SSIM of a candidate video vs a reference")
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--output", default="outputs/comparison")
    p.add_argument("--max-frames", type=int)

    p = sub.add_parser("complexity", help="Spatial/temporal complexity and scene cuts (uses C++ kernels)")
    p.add_argument("input")
    p.add_argument("--output", default="outputs/complexity")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--no-native", action="store_true", help="force the pure-Python reference kernels")
    p.add_argument("--method", choices=["diamond", "full"], default="diamond")

    p = sub.add_parser("sweep", help="x264 CRF sweep: bitrate vs quality + recommended CRF")
    p.add_argument("input")
    p.add_argument("--output", default="outputs/sweep")
    p.add_argument("--crfs", default="18,20,22,24,26,28,30,32,34")
    p.add_argument("--preset", default="medium")
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--metric", choices=["auto", "vmaf", "ssim", "psnr"], default="auto")
    p.add_argument("--target", type=float, help="quality target (default: vmaf 93, ssim 0.95, psnr 38)")

    p = sub.add_parser("kernel-benchmark", help="Brightness/contrast: NumPy vs OpenCV LUT vs C++ LUT")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--repeats", type=int, default=20)

    p = sub.add_parser("motion-benchmark", help="Motion estimation: Python vs C++ (diamond/full search)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--include-python-full", action="store_true", help="slow: also time the Python full search")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "info":
            caps = capabilities()
            _dump({"framelab": __version__, "native_module": kernels.native_available(), **caps})
        elif args.command == "inspect":
            r = inspect_video(args.input, args.output, args.max_frames)
            r.pop("frame_metrics", None)
            _dump(r)
        elif args.command == "process":
            r = process_video(args.input, args.output, args.method, max_frames=args.max_frames,
                              analyze=not args.no_analyze, backend=args.backend, crf=args.crf)
            r.pop("frame_metrics", None)
            _dump(r)
        elif args.command == "compare":
            r = compare_videos(args.reference, args.candidate, args.output, args.max_frames)
            r.pop("frame_metrics", None)
            _dump(r)
        elif args.command == "complexity":
            from pathlib import Path
            summary, df = analyze_complexity(args.input, args.max_frames, method=args.method,
                                             prefer_native=not args.no_native)
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(out / "complexity.csv", index=False)
            summary["csv"] = str(out / "complexity.csv")
            _dump(summary)
        elif args.command == "sweep":
            crfs = [int(x) for x in args.crfs.split(",") if x.strip()]
            r = crf_sweep(args.input, args.output, crfs, args.preset, args.max_frames, args.metric, args.target)
            r.pop("table", None)
            _dump(r)
        elif args.command == "kernel-benchmark":
            _dump(kernel_benchmark(args.width, args.height, args.repeats))
        elif args.command == "motion-benchmark":
            _dump(motion_benchmark(args.width, args.height, args.repeats,
                                   include_python_full=args.include_python_full))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
