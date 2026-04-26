from typing import Dict

import numpy as np
from scipy.signal import medfilt

from .config import CFG


def correct_harmonic_jumps(
    pred: np.ndarray,
    hr_min: float = 40.0,
    hr_max: float = 200.0,
    window: int = 15,
    thresh_jump: float = 18.0,
) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float32).copy()

    for i in range(len(out)):
        left = max(0, i - window)
        hist = out[left:i]

        if hist.size < max(3, window // 2):
            continue

        ref = float(np.median(hist))
        cur = float(out[i])

        half = cur / 2.0
        double = cur * 2.0

        if abs(cur - ref) > thresh_jump and hr_min <= half <= hr_max:
            if abs(half - ref) < 0.60 * abs(cur - ref):
                out[i] = half
                continue

        if abs(cur - ref) > thresh_jump and hr_min <= double <= hr_max:
            if abs(double - ref) < 0.60 * abs(cur - ref):
                out[i] = double
                continue

        if abs(cur - ref) > 40.0:
            out[i] = np.clip(ref, hr_min, hr_max)

    return out.astype(np.float32)


def temporal_smooth(pred: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)

    if len(pred) < kernel_size or kernel_size < 3:
        return pred.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1

    return medfilt(pred, kernel_size=kernel_size).astype(np.float32)


def apply_quality_gate(
    pred: np.ndarray,
    sigma: np.ndarray,
    quality: np.ndarray,
    cfg: CFG,
) -> Dict[str, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    quality = np.asarray(quality, dtype=np.float32)

    out = pred.copy()
    flags = np.asarray(["accept"] * len(pred), dtype=object)

    if len(pred) == 0:
        return {
            "pred": out,
            "flags": flags,
            "kept_mask": np.asarray([], dtype=bool),
        }

    prev = float(out[0])

    for i in range(len(out)):
        q = float(quality[i])
        s = float(sigma[i])
        jump = abs(float(out[i]) - prev)

        uncertain = (q < 0.35) or (s > cfg.retained_sigma_max) or (jump > 18.0)

        if uncertain and i > 0:
            alpha = 0.35
            out[i] = alpha * float(out[i]) + (1.0 - alpha) * prev
            flags[i] = "uncertain"

        out[i] = np.clip(out[i], cfg.hr_min_bpm, cfg.hr_max_bpm)
        prev = float(out[i])

    kept_mask = flags != "uncertain"

    return {
        "pred": out.astype(np.float32),
        "flags": flags,
        "kept_mask": kept_mask,
    }


def postprocess_predictions(
    pred_raw: np.ndarray,
    sigma: np.ndarray,
    quality: np.ndarray,
    cfg: CFG,
) -> Dict[str, np.ndarray]:
    pred = np.asarray(pred_raw, dtype=np.float32)

    if cfg.medfilt_k >= 3:
        pred = temporal_smooth(pred, kernel_size=cfg.medfilt_k)

    pred = correct_harmonic_jumps(
        pred,
        hr_min=cfg.hr_min_bpm,
        hr_max=cfg.hr_max_bpm,
    )

    info = apply_quality_gate(pred, sigma=sigma, quality=quality, cfg=cfg)

    return info
