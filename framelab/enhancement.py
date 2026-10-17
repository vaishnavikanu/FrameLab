"""Image enhancement algorithms used by FrameLab."""
import cv2
import numpy as np

from .kernels import adjust_brightness_contrast, adjust_brightness_contrast_numpy  # re-exported

METHODS = ("identity", "clahe", "denoise", "sharpen", "combined", "brightness")


def _validate(frame):
    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty image")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _clahe_luma(frame, clip_limit, tile_grid_size):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def _denoise(frame, strength):
    return cv2.fastNlMeansDenoisingColored(frame, None, float(strength), float(strength), 7, 21)


def apply_enhancement(frame, method="identity", clip_limit=2.0, tile_grid_size=8,
                      denoise_strength=5, sharpen_amount=0.7, alpha=1.15, beta=10.0):
    frame = _validate(frame)
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if method == "identity":
        return frame.copy()
    if method == "clahe":
        return _clahe_luma(frame, clip_limit, tile_grid_size)
    if method == "denoise":
        return _denoise(frame, denoise_strength)
    if method == "sharpen":
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
        return cv2.addWeighted(frame, 1.0 + float(sharpen_amount), blurred, -float(sharpen_amount), 0)
    if method == "brightness":  # uses the native LUT kernel when it has been built
        return adjust_brightness_contrast(frame, alpha, beta)
    # combined: denoise, then local contrast, then mild sharpening
    contrasted = _clahe_luma(_denoise(frame, denoise_strength), clip_limit, tile_grid_size)
    blurred = cv2.GaussianBlur(contrasted, (0, 0), 1.0)
    return cv2.addWeighted(contrasted, 1.35, blurred, -0.35, 0)
