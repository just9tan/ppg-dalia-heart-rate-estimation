import random
from typing import Tuple

import numpy as np
import torch


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nan_to_num_np(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def robust_zscore_1d(x: np.ndarray, eps: float = 1e-6, clip: float = 5.0) -> np.ndarray:
    x = nan_to_num_np(np.asarray(x, dtype=np.float32).reshape(-1))
    x = x - np.mean(x, dtype=np.float64)
    std = float(np.std(x, dtype=np.float64))

    if not np.isfinite(std) or std < eps:
        std = 1.0

    x = x / std

    if clip is not None and clip > 0:
        x = np.clip(x, -clip, clip)

    return x.astype(np.float32)


def robust_zscore_2d(x: np.ndarray, eps: float = 1e-6, clip: float = 5.0) -> np.ndarray:
    x = nan_to_num_np(np.asarray(x, dtype=np.float32))

    if x.ndim == 1:
        return robust_zscore_1d(x, eps=eps, clip=clip)[:, None]

    out = np.zeros_like(x, dtype=np.float32)

    for c in range(x.shape[1]):
        out[:, c] = robust_zscore_1d(x[:, c], eps=eps, clip=clip)

    return out.astype(np.float32)


def extract_1d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    x = np.asarray(x)
    end = start + length

    if start < 0:
        left = np.zeros((-start,), dtype=x.dtype)
        seg = np.concatenate([left, x[:end]], axis=0)
    else:
        seg = x[start:end]

    if seg.size < length:
        seg = np.concatenate([seg, np.zeros(length - seg.size, dtype=x.dtype)], axis=0)

    return seg


def extract_2d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim == 1:
        x = x[:, None]

    end = start + length

    if start < 0:
        left = np.zeros((-start, x.shape[1]), dtype=x.dtype)
        seg = np.concatenate([left, x[:end]], axis=0)
    else:
        seg = x[start:end]

    if seg.shape[0] < length:
        pad = np.zeros((length - seg.shape[0], x.shape[1]), dtype=x.dtype)
        seg = np.concatenate([seg, pad], axis=0)

    return seg


def resample_linear_np(x: np.ndarray, out_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        xp = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
        xnew = np.linspace(0.0, 1.0, num=out_len, dtype=np.float32)
        return np.interp(xnew, xp, x).astype(np.float32)

    y = np.zeros((out_len, x.shape[1]), dtype=np.float32)

    for c in range(x.shape[1]):
        y[:, c] = resample_linear_np(x[:, c], out_len)

    return y.astype(np.float32)


def ensure_three_axis_acc(acc: np.ndarray) -> np.ndarray:
    acc = np.asarray(acc, dtype=np.float32)

    if acc.ndim == 1:
        acc = acc[:, None]

    if acc.shape[1] == 3:
        return acc.astype(np.float32)

    out = np.zeros((acc.shape[0], 3), dtype=np.float32)
    out[:, : min(3, acc.shape[1])] = acc[:, : min(3, acc.shape[1])]

    return out


def build_lag_matrix(x: np.ndarray, lags: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        x = x[:, None]

    T, C = x.shape
    feats = []

    for lag in range(max(1, int(lags))):
        if lag == 0:
            feats.append(x)
        else:
            pad = np.zeros((lag, C), dtype=np.float32)
            feats.append(np.concatenate([pad, x[:-lag]], axis=0))

    return np.concatenate(feats, axis=1).astype(np.float32)


def filter_ppg_window(
    ppg: np.ndarray,
    acc_up: np.ndarray,
    lags: int = 3,
    ridge: float = 1e-2,
    use_derivative: bool = True,
) -> np.ndarray:
    """
    Generic ACC-aware PPG filtering for public release.

    This is intentionally written as a general preprocessing step.
    Exact internal experimental variants and tuning values are not included.
    """
    ppg = robust_zscore_1d(ppg)
    acc_up = robust_zscore_2d(acc_up)

    X = build_lag_matrix(acc_up, lags=lags)

    if use_derivative:
        dacc = np.diff(acc_up, axis=0, prepend=acc_up[:1])
        X = np.concatenate([X, build_lag_matrix(dacc, lags=max(1, lags - 1))], axis=1)

    XtX = X.T @ X
    XtY = X.T @ ppg
    XtX = XtX + float(ridge) * np.eye(XtX.shape[0], dtype=np.float32)

    try:
        w = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(XtX, XtY, rcond=None)[0]

    estimated_artifact = X @ w
    filtered_ppg = ppg - estimated_artifact

    return robust_zscore_1d(filtered_ppg)


def compute_window_quality(ppg: np.ndarray, acc: np.ndarray) -> Tuple[float, float]:
    """
    Lightweight public quality estimate.

    Returns:
        ppg_energy, acc_energy
    """
    ppg = np.asarray(ppg, dtype=np.float32).reshape(-1)
    acc = ensure_three_axis_acc(acc)

    ppg_energy = float(np.sqrt(np.mean(ppg * ppg) + 1e-8))
    acc_energy = float(np.sqrt(np.mean(np.sum(acc * acc, axis=1)) + 1e-8))

    return ppg_energy, acc_energy
