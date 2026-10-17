import numpy as np
import pytest

from framelab.analysis import analyze_frame
from framelab.enhancement import METHODS, apply_enhancement
from framelab.metrics import psnr, ssim


@pytest.fixture
def frame():
    return np.random.default_rng(4).integers(0, 256, (96, 128, 3), dtype=np.uint8)


@pytest.mark.parametrize("method", METHODS)
def test_enhancement_returns_same_shape_uint8(frame, method):
    result = apply_enhancement(frame, method)
    assert result.shape == frame.shape and result.dtype == np.uint8


def test_identity_copies_frame(frame):
    result = apply_enhancement(frame, "identity")
    assert np.array_equal(result, frame) and result is not frame


def test_unknown_method_and_empty_frame_rejected(frame):
    with pytest.raises(ValueError):
        apply_enhancement(frame, "magic")
    with pytest.raises(ValueError):
        apply_enhancement(np.zeros((0, 0, 3), np.uint8), "identity")


def test_denoise_reduces_noise_proxy_and_sharpen_raises_blur_proxy():
    import cv2
    rng = np.random.default_rng(1)
    clean = cv2.GaussianBlur(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8), (0, 0), 4)
    noisy = np.clip(clean + rng.normal(0, 8, clean.shape), 0, 255).astype(np.uint8)
    assert analyze_frame(apply_enhancement(noisy, "denoise", denoise_strength=10))["noise_residual_std"] \
        < analyze_frame(noisy)["noise_residual_std"]
    assert analyze_frame(apply_enhancement(clean, "sharpen"))["blur_laplacian_variance"] \
        > analyze_frame(clean)["blur_laplacian_variance"]


def test_brightness_method_brightens(frame):
    assert analyze_frame(apply_enhancement(frame, "brightness", alpha=1.0, beta=20))["brightness_mean"] \
        > analyze_frame(frame)["brightness_mean"]


def test_analysis_has_expected_keys(frame):
    result = analyze_frame(frame)
    assert set(result) == {"brightness_mean", "blur_laplacian_variance", "noise_residual_std"}
    assert all(np.isfinite(v) for v in result.values())


def test_metrics_identical_and_known_psnr(frame):
    assert psnr(frame, frame) == float("inf")
    assert ssim(frame, frame) == pytest.approx(1.0, abs=1e-6)
    shifted = np.clip(frame.astype(int) + 10, 0, 255).astype(np.uint8)
    bright = np.full((8, 8, 3), 100, np.uint8)
    assert psnr(bright, np.full_like(bright, 110)) == pytest.approx(10 * np.log10(255 ** 2 / 100), abs=1e-9)
    assert ssim(frame, shifted) < 1.0


def test_ssim_matches_scikit_image():
    skimage = pytest.importorskip("skimage.metrics")
    import cv2
    rng = np.random.default_rng(2)
    a = cv2.GaussianBlur(rng.integers(0, 256, (120, 160), dtype=np.uint8), (0, 0), 3)
    b = np.clip(a + rng.normal(0, 10, a.shape), 0, 255).astype(np.uint8)
    ref = skimage.structural_similarity(a, b, gaussian_weights=True, sigma=1.5,
                                        use_sample_covariance=False, data_range=255)
    assert ssim(a, b) == pytest.approx(ref, abs=2e-3)
