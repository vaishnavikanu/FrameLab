"""Generate a synthetic test video (no downloads, no copyright): samples/sample.mp4

Two scenes joined by a hard cut, 640x360 @ 25 fps, 6 seconds:
  scene 1 - dark, mildly noisy, panning texture + moving box   (good for CLAHE / denoise demos)
  scene 2 - bright, sharper texture + moving circle      (a clear scene cut for the complexity tool)
A 440 Hz tone is added as audio when ffmpeg is available, so audio preservation can be verified.

Usage:  python scripts/make_sample_video.py [output_path]
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framelab.ffmpeg_tools import find_ffmpeg, have_x264  # noqa: E402

W, H, FPS, PER_SCENE = 640, 360, 25, 75


def texture(seed, sigma, w=1400, h=900):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.integers(0, 256, (h, w), dtype=np.uint8), (0, 0), sigma)
    return cv2.cvtColor(cv2.normalize(t, None, 40, 230, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)


def frames():
    rng = np.random.default_rng(7)
    t1, t2 = texture(1, 3.0), texture(2, 1.6)
    for i in range(PER_SCENE):                                     # scene 1: dark + noisy
        x, y = 100 + 3 * i, 120 + i
        f = (t1[y:y + H, x:x + W].astype(np.float32) * 0.35)
        cv2.rectangle(f, (40 + 4 * i, 150), (120 + 4 * i, 210), (60, 110, 200), -1)
        yield np.clip(f + rng.normal(0, 2.5, f.shape), 0, 255).astype(np.uint8)
    for i in range(PER_SCENE):                                     # scene 2: bright + sharp
        x, y = 700 - 4 * i, 300 + 2 * i
        f = t2[y:y + H, x:x + W].copy()
        cv2.circle(f, (60 + 6 * i, 180 + int(40 * np.sin(i / 6))), 34, (30, 200, 240), -1)
        yield f


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "samples" / "sample.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if have_x264():
        seconds = 2 * PER_SCENE / FPS
        cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
               "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames():
            p.stdin.write(f.tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit("ffmpeg failed while writing the sample video")
    else:
        w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames():
            w.write(f)
        w.release()
        print("ffmpeg/libx264 not found: wrote a silent mp4v video instead.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
