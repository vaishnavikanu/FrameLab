import cv2
import numpy as np
import pytest

from framelab import kernels as k

needs_native = pytest.mark.skipif(not k.native_available(),
                                  reason="C++ module not built (python setup.py build_ext --inplace)")


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(0).integers(0, 256, (50, 70, 3), dtype=np.uint8)


ALPHAS = (0.0, 0.3, 1.0, 1.1, 1.15, 1.7, 2.0, 3.3)
BETAS = (-40.0, -0.5, 0.0, 0.5, 8.0, 10.0, 200.0)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("beta", BETAS)
def test_lut_matches_numpy_reference(img, alpha, beta):
    ref = k.adjust_brightness_contrast_numpy(img, alpha, beta)
    assert np.array_equal(ref, k.adjust_brightness_contrast_lut(img, alpha, beta))
    assert np.array_equal(ref, k.adjust_brightness_contrast(img, alpha, beta, prefer_native=False))


@needs_native
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("beta", BETAS)
def test_native_brightness_is_bit_exact(img, alpha, beta):
    assert np.array_equal(k.adjust_brightness_contrast_numpy(img, alpha, beta),
                          k.adjust_brightness_contrast(img, alpha, beta, prefer_native=True))


def test_brightness_clips_and_rejects_non_uint8():
    image = np.array([[[0, 100, 255]]], dtype=np.uint8)
    assert k.adjust_brightness_contrast(image, 2.0, 10).tolist() == [[[10, 210, 255]]]
    with pytest.raises(ValueError):
        k.adjust_brightness_contrast(image.astype(np.float32), 1.0, 0.0)


def _pair(shift_rows=2, shift_cols=-3, h=96, w=128):
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.integers(0, 256, (h, w), dtype=np.uint8), (0, 0), 2.0)
    return base, np.roll(base, (shift_rows, shift_cols), (0, 1))


def test_full_search_recovers_planted_motion_python():
    prev, cur = _pair()
    mvs, sads = k.motion_estimate(prev, cur, 16, 8, "full", prefer_native=False)
    inner_mv, inner_sad = mvs[1:, :-1], sads[1:, :-1]   # blocks whose source did not wrap around
    assert (inner_mv == np.array([3, -2])).all() and (inner_sad == 0).all()


@needs_native
@pytest.mark.parametrize("method", ["diamond", "full"])
def test_native_motion_equals_python_reference(method):
    prev, cur = _pair(h=64, w=96)
    a = k.motion_estimate(prev, cur, 8, 6, method, prefer_native=True)
    b = k.motion_estimate(prev, cur, 8, 6, method, prefer_native=False)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@needs_native
def test_native_motion_recovers_motion_and_diamond_is_close_to_full():
    prev, cur = _pair()
    mv_full, sad_full = k.motion_estimate(prev, cur, 16, 8, "full")
    mv_dia, sad_dia = k.motion_estimate(prev, cur, 16, 8, "diamond")
    assert (mv_full[1:, :-1] == np.array([3, -2])).all()
    assert (sad_dia >= sad_full).all()                       # full search is optimal
    assert (sad_dia[1:, :-1] == 0).mean() >= 0.7             # diamond finds most exact matches


def test_zero_motion_on_identical_frames():
    prev, _ = _pair()
    mvs, sads = k.motion_estimate(prev, prev, 16, 8, "diamond")
    assert (mvs == 0).all() and (sads == 0).all()


@pytest.mark.parametrize("prefer_native", [True, False])
def test_motion_input_validation(prefer_native):
    a = np.zeros((32, 32), np.uint8)
    with pytest.raises((ValueError, RuntimeError)):
        k.motion_estimate(a, np.zeros((32, 48), np.uint8), 16, 8, "diamond", prefer_native=prefer_native)
    with pytest.raises((ValueError, RuntimeError)):
        k.motion_estimate(a, a, 64, 8, "diamond", prefer_native=prefer_native)
    with pytest.raises((ValueError, RuntimeError)):
        k.motion_estimate(a, a, 16, 8, "nope", prefer_native=prefer_native)


def test_gradient_energy_known_values():
    flat = np.full((10, 10), 7, np.uint8)
    assert k.gradient_energy(flat) == 0.0
    ramp = np.tile(np.arange(10, dtype=np.uint8), (10, 1))      # dx=1 everywhere, dy=0
    assert k.gradient_energy_python(ramp) == pytest.approx(1.0)
    assert k.gradient_energy(ramp) == pytest.approx(1.0)


@needs_native
def test_native_gradient_energy_matches_python():
    g = np.random.default_rng(5).integers(0, 256, (61, 83), dtype=np.uint8)
    assert k.gradient_energy(g, prefer_native=True) == k.gradient_energy_python(g)
