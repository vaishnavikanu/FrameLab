"""Video metadata, decoding, and writer selection."""
from pathlib import Path

import cv2

from .ffmpeg_tools import FFmpegError, FFmpegWriter, have_x264


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {path}. Try an MP4 supported by your OpenCV/FFmpeg build.")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps != fps:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, {"fps": fps, "width": width, "height": height,
                 "frame_count_reported": count,
                 "duration_seconds_reported": (count / fps if count > 0 else None)}


class CVWriter:
    """OpenCV fallback writer (mpeg4/`mp4v`; no audio, and browsers often cannot play it)."""

    def __init__(self, path, fps, width, height):
        path = Path(path)
        ext = path.suffix.lower()
        codecs = ["mp4v", "avc1"] if ext == ".mp4" else ["XVID", "MJPG"]
        self._w = None
        for codec in codecs:
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps),
                                (int(width), int(height)))
            if w.isOpened():
                self._w, self.backend, self.audio = w, f"opencv-{codec}", False
                return
            w.release()
        raise RuntimeError(f"Could not create an OpenCV video writer for {path}.")

    def write(self, frame):
        self._w.write(frame)

    def release(self):
        self._w.release()


def create_writer(path, fps, width, height, backend="auto", crf=18, preset="medium", audio_source=None):
    """backend: 'auto' (ffmpeg/x264 if available, else OpenCV), 'ffmpeg', or 'opencv'."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend not in ("auto", "ffmpeg", "opencv"):
        raise ValueError("backend must be 'auto', 'ffmpeg' or 'opencv'")
    if backend in ("auto", "ffmpeg") and have_x264():
        return FFmpegWriter(path, fps, width, height, crf=crf, preset=preset, audio_source=audio_source)
    if backend == "ffmpeg":
        raise FFmpegError("ffmpeg with libx264 is not available.")
    return CVWriter(path, fps, width, height)
