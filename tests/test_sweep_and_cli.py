import json

import pytest

from framelab.cli import main
from framelab.ffmpeg_tools import compute_vmaf, have_vmaf, have_x264
from framelab.sweep import crf_sweep

needs_x264 = pytest.mark.skipif(not have_x264(), reason="needs ffmpeg with libx264")


@needs_x264
def test_sweep_is_monotonic_and_recommends_correctly(make_video, tmp_path):
    src = make_video(n=10, noise=6.0)
    r = crf_sweep(src, tmp_path / "sw", crfs=(18, 28, 38), preset="ultrafast", max_frames=10,
                  metric="ssim", target=0.0)
    t = r["table"]
    assert [x["crf"] for x in t] == [18, 28, 38]
    assert t[0]["size_kb"] > t[1]["size_kb"] > t[2]["size_kb"]
    assert t[0]["mean_ssim"] > t[1]["mean_ssim"] > t[2]["mean_ssim"]
    assert r["recommended_crf"] == 38                       # target 0 -> cheapest passes
    assert (tmp_path / "sw" / "rd_curve.png").exists() and (tmp_path / "sw" / "crf_sweep.csv").exists()
    r2 = crf_sweep(src, tmp_path / "sw2", crfs=(28, 38), preset="ultrafast", max_frames=10,
                   metric="ssim", target=0.99999, with_complexity=False)
    assert r2["recommended_crf"] is None


@needs_x264
def test_sweep_rejects_vmaf_when_unavailable(make_video, tmp_path):
    if have_vmaf():
        pytest.skip("this ffmpeg has libvmaf")
    with pytest.raises(RuntimeError):
        crf_sweep(make_video(n=4), tmp_path / "o", crfs=(20,), metric="vmaf")


@pytest.mark.skipif(not have_vmaf(), reason="ffmpeg without libvmaf")
def test_vmaf_of_identical_video_is_about_100(make_video, tmp_path):
    src = make_video(n=6)
    assert compute_vmaf(src, src, work_dir=tmp_path / "v") > 99.0


def test_cli_info_and_errors(capsys):
    assert main(["info"]) == 0
    assert "native_module" in json.loads(capsys.readouterr().out)
    assert main(["inspect", "does_not_exist.mp4"]) == 1


def test_cli_kernel_benchmark_runs(capsys):
    assert main(["kernel-benchmark", "--width", "64", "--height", "48", "--repeats", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert all(out["identical_to_numpy"].values())
