"""Reference-based image quality metrics (own implementations, no scikit-image needed)."""
import cv2
import numpy as np


def mse(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes")
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(reference, candidate):
    error = mse(reference, candidate)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / error))


def _gray64(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)


def ssim(reference, candidate):
    """Mean SSIM on luma (11x11 Gaussian window, sigma 1.5, K1=0.01, K2=0.03).

    Verified against scikit-image's gaussian-weighted SSIM to ~3 decimal places.
    Computed in float64 end to end to avoid cancellation error in the variance terms.
    """
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes")
    x, y = _gray64(reference), _gray64(candidate)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(a):
        return cv2.GaussianBlur(a, (11, 11), 1.5)

    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x * mu_x
    syy = blur(y * y) - mu_y * mu_y
    sxy = blur(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))
