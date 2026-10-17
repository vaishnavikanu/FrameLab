import numpy as np
import pytest

from framelab import kernels
from framelab.complexity import analyze_complexity
from conftest import textured_frames, write_video


def test_static_video_has_no_residual_and_no_cuts(tmp_path):
    frame = textured_frames(1)[0]
    src = write_video(tmp_path / "static.mp4", [frame] * 10)
    s, df = analyze_complexity(src, analysis_width=160)
    assert s["scene_cuts"] == 0 and s["mean_temporal_residual"] < 1.5 and len(df) == 10


def test_scene_cut_is_detected_at_the_right_frame(tmp_path):
    a = textured_frames(8, seed=1, shift=(2, 0))
    b = textured_frames(8, seed=99, shift=(2, 0))
    src = write_video(tmp_path / "cut.mp4", a + b)
    s, df = analyze_complexity(src, analysis_width=160)
    assert s["scene_cut_frames"] == [8]


def test_busier_content_scores_higher(tmp_path):
    rng = np.random.default_rng(0)
    smooth = write_video(tmp_path / "smooth.mp4", textured_frames(8, shift=(1, 0)))
    noisy_frames = [np.clip(f.astype(int) + rng.normal(0, 25, f.shape), 0, 255).astype(np.uint8)
                    for f in textured_frames(8, shift=(1, 0))]
    noisy = write_video(tmp_path / "noisy.mp4", noisy_frames)
    assert analyze_complexity(noisy, analysis_width=160)[0]["complexity_index"] \
        > analyze_complexity(smooth, analysis_width=160)[0]["complexity_index"]


@pytest.mark.skipif(not kernels.native_available(), reason="C++ module not built")
def test_native_and_python_kernels_give_same_summary(make_video):
    src = make_video(n=6)
    a, _ = analyze_complexity(src, analysis_width=160, prefer_native=True)
    b, _ = analyze_complexity(src, analysis_width=160, prefer_native=False)
    for key in ("mean_spatial_complexity", "mean_temporal_residual", "mean_motion_pixels", "scene_cuts"):
        assert a[key] == pytest.approx(b[key])
