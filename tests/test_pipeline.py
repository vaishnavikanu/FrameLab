import numpy as np
import pytest

from framelab.ffmpeg_tools import find_ffmpeg, has_audio, have_x264, run_ffmpeg
from framelab.pipeline import compare_videos, inspect_video, process_video


def test_process_video_end_to_end(make_video, tmp_path):
    src = make_video(n=8)
    out = tmp_path / "out.mp4"
    r = process_video(src, out, method="clahe")
    assert r["frames_processed"] == 8
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / "out.csv").exists()
    assert r["mean_enhance_ms"] > 0 and r["output_bitrate_kbps"] > 0
    assert "mean_noise_residual_std_in" in r and "mean_psnr_vs_input_db" in r


def test_identity_processing_reports_100db_vs_input(make_video, tmp_path):
    src = make_video(n=5)
    r = process_video(src, tmp_path / "o.mp4", method="identity")
    assert r["mean_psnr_vs_input_db"] == 100.0
    assert r["mean_blur_laplacian_variance_in"] == pytest.approx(r["mean_blur_laplacian_variance_out"])


def test_max_frames_and_no_analyze(make_video, tmp_path):
    src = make_video(n=10)
    r = process_video(src, tmp_path / "o.mp4", method="sharpen", max_frames=4, analyze=False)
    assert r["frames_processed"] == 4
    assert "mean_psnr_vs_input_db" not in r


def test_opencv_backend_fallback_works(make_video, tmp_path):
    src = make_video(n=5)
    try:
        r = process_video(src, tmp_path / "cv.mp4", method="identity", backend="opencv")
    except RuntimeError:
        pytest.skip("OpenCV build cannot write mp4v here")
    assert r["frames_processed"] == 5 and r["writer_backend"].startswith("opencv") and not r["audio_preserved"]


@pytest.mark.skipif(not have_x264(), reason="needs ffmpeg with libx264")
def test_audio_is_preserved(tmp_path):
    src = tmp_path / "av.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=160x96:rate=10:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src])
    assert has_audio(src)
    out = tmp_path / "o.mp4"
    r = process_video(src, out, method="clahe")
    assert r["audio_preserved"] and has_audio(out)


@pytest.mark.skipif(not have_x264(), reason="needs ffmpeg with libx264")
def test_odd_dimensions_are_handled(tmp_path):
    src = tmp_path / "odd.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=162x98:rate=10:duration=1",
                "-vf", "crop=161:97", "-c:v", "libx264", "-pix_fmt", "yuv444p", src])
    r = process_video(src, tmp_path / "o.mp4", method="identity")
    assert r["frames_processed"] == 10 and r["output_size_bytes"] > 0


def test_inspect_and_compare(make_video, tmp_path):
    src = make_video(n=6)
    ins = inspect_video(src, tmp_path / "ins")
    assert ins["frames_processed"] == 6 and (tmp_path / "ins" / "inspection.csv").exists()
    cmp = compare_videos(src, src, tmp_path / "cmp")
    assert cmp["frames_compared"] == 6 and cmp["mean_psnr_db"] == 100.0
    assert cmp["mean_ssim"] == pytest.approx(1.0, abs=1e-6) and not cmp["length_mismatch"]


def test_compare_flags_length_mismatch(make_video, tmp_path):
    a, b = make_video("a.mp4", n=6), make_video("b.mp4", n=9)
    r = compare_videos(a, b, tmp_path / "c")
    assert r["frames_compared"] == 6 and r["length_mismatch"]


def test_bad_inputs_raise(tmp_path):
    with pytest.raises(ValueError):
        process_video(tmp_path / "missing.mp4", tmp_path / "o.mp4")
