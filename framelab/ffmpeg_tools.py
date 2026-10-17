"""Thin, defensive wrappers around the ffmpeg binary.

ffmpeg is found in this order: $FFMPEG_BINARY, `ffmpeg` on PATH, then the static binary shipped
by the `imageio-ffmpeg` pip package (so `pip install -r requirements.txt` is enough on Windows).
Everything degrades gracefully when ffmpeg or libvmaf is missing.
"""
import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def find_ffmpeg():
    env = os.environ.get("FFMPEG_BINARY")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run(args, cwd=None):
    return subprocess.run(args, capture_output=True, text=True, cwd=cwd,
                          encoding="utf-8", errors="replace")


@functools.lru_cache(maxsize=1)
def capabilities():
    exe = find_ffmpeg()
    caps = {"ffmpeg": exe, "version": None, "libx264": False, "libvmaf": False}
    if not exe:
        return caps
    first = _run([exe, "-hide_banner", "-version"]).stdout.splitlines()
    caps["version"] = first[0] if first else None
    caps["libx264"] = "libx264" in _run([exe, "-hide_banner", "-encoders"]).stdout
    caps["libvmaf"] = "libvmaf" in _run([exe, "-hide_banner", "-filters"]).stdout
    return caps


def have_x264():
    return bool(capabilities()["libx264"])


def have_vmaf():
    return bool(capabilities()["libvmaf"])


EVEN_CROP = "crop=trunc(iw/2)*2:trunc(ih/2)*2"  # yuv420p needs even dimensions


def run_ffmpeg(args, cwd=None):
    exe = find_ffmpeg()
    if not exe:
        raise FFmpegError("ffmpeg not found. Run `pip install imageio-ffmpeg` or install ffmpeg.")
    proc = _run([exe, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], cwd=cwd)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
    return proc


def has_audio(path):
    exe = find_ffmpeg()
    if not exe:
        return False
    out = _run([exe, "-hide_banner", "-i", str(path)]).stderr  # exits 1 without output: expected
    return bool(re.search(r"Stream #\d+:\d+.*Audio:", out))


def encode_x264(input_path, output_path, crf=23, preset="medium", max_frames=None, keep_audio=False):
    """Encode `input_path` with libx264 at constant quality. Returns the output path."""
    if not have_x264():
        raise FFmpegError("This ffmpeg build has no libx264 encoder.")
    args = ["-i", input_path, "-vf", EVEN_CROP]
    if max_frames:
        args += ["-frames:v", int(max_frames)]
    args += ["-c:v", "libx264", "-preset", preset, "-crf", int(crf), "-pix_fmt", "yuv420p"]
    args += ["-c:a", "aac", "-b:a", "128k"] if keep_audio else ["-an"]
    args += [output_path]
    run_ffmpeg(args)
    return output_path


def compute_vmaf(distorted, reference, max_frames=None, work_dir=None):
    """Mean VMAF of `distorted` against `reference`, or None if libvmaf is unavailable.

    Both inputs are cropped to even dimensions and trimmed to `max_frames` so they line up.
    """
    if not have_vmaf():
        return None
    work = Path(work_dir or tempfile.mkdtemp(prefix="framelab_vmaf_"))
    work.mkdir(parents=True, exist_ok=True)
    log_name = "vmaf_log.json"
    trim = f",trim=end_frame={int(max_frames)}" if max_frames else ""
    graph = (f"[0:v]{EVEN_CROP}{trim},setpts=PTS-STARTPTS[d];"
             f"[1:v]{EVEN_CROP}{trim},setpts=PTS-STARTPTS[r];"
             f"[d][r]libvmaf=log_fmt=json:log_path={log_name}:n_threads=4")
    # cwd + relative log path avoids Windows drive-letter colon escaping problems in filter graphs
    run_ffmpeg(["-i", Path(distorted).resolve(), "-i", Path(reference).resolve(),
                "-lavfi", graph, "-f", "null", "-"], cwd=work)
    data = json.loads((work / log_name).read_text(encoding="utf-8"))
    return float(data["pooled_metrics"]["vmaf"]["mean"])


class FFmpegWriter:
    """Pipe raw BGR frames into ffmpeg -> H.264 MP4, optionally muxing the source audio."""
    backend = "ffmpeg-x264"

    def __init__(self, path, fps, width, height, crf=18, preset="medium", audio_source=None):
        if not have_x264():
            raise FFmpegError("This ffmpeg build has no libx264 encoder.")
        self.path = str(path)
        self.audio = bool(audio_source and has_audio(audio_source))
        args = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{int(width)}x{int(height)}",
                "-r", f"{float(fps):.6f}", "-i", "-"]
        if self.audio:
            args += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac",
                     "-b:a", "128k", "-shortest"]
        args += ["-vf", EVEN_CROP, "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path]
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                      stderr=self._err)

    def write(self, frame):
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            raise FFmpegError(f"ffmpeg stopped early: {self._stderr()}")

    def _stderr(self):
        self._err.seek(0)
        return self._err.read().decode("utf-8", "replace").strip()[-800:]

    def release(self):
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except OSError:
            pass
        code = self._proc.wait()
        msg = self._stderr()
        self._err.close()
        if code != 0:
            raise FFmpegError(f"ffmpeg exited with {code}: {msg}")
