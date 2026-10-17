"""Lightweight, explainable frame-quality indicators. These are proxies, not ground truth."""
import cv2
import numpy as np


def analyze_frame(frame):
    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty image")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray_f = gray.astype(np.float32)
    brightness = float(np.mean(gray_f))
    blur_proxy = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_proxy = float(np.std(gray_f - smooth.astype(np.float32)))
    return {
        "brightness_mean": brightness,
        "blur_laplacian_variance": blur_proxy,
        "noise_residual_std": noise_proxy,
    }
