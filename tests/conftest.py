import cv2
import numpy as np
import pytest

from framelab.ffmpeg_tools import FFmpegWriter, have_x264


def textured_frames(n=12, w=160, h=96, seed=0, shift=(3, 1), noise=0.0):
    """Frames of a panning blurred-noise texture: real, recoverable motion."""
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.integers(0, 256, (h + 80, w + 80 + n * 8), dtype=np.uint8), (0, 0), 2.0)
    out = []
    for i in range(n):
        x, y = 10 + shift[0] * i, 10 + shift[1] * i
        g = big[y:y + h, x:x + w].astype(np.float32)
        if noise:
            g = g + rng.normal(0, noise, g.shape)
        out.append(cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return out


def write_video(path, frames, fps=10):
    """Write frames with x264 if available, else OpenCV; skip the test if neither can write."""
    h, w = frames[0].shape[:2]
    if have_x264():
        wr = FFmpegWriter(path, fps, w, h, crf=14, preset="ultrafast")
    else:
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not wr.isOpened():
            pytest.skip("no usable video encoder in this environment")
    for f in frames:
        wr.write(f)
    wr.release()
    return path


@pytest.fixture
def make_video(tmp_path):
    def _make(name="v.mp4", **kw):
        fps = kw.pop("fps", 10)
        return write_video(tmp_path / name, textured_frames(**kw), fps=fps)
    return _make
