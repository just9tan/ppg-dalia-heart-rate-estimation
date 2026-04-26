from typing import Dict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from .config import CFG
from .network import HeartRateEstimator
from .inference import postprocess_predictions


@torch.inference_mode()
def evaluate_loader(
    model: HeartRateEstimator,
    loader: DataLoader,
    device: torch.device,
    cfg: CFG,
) -> Dict[str, object]:
    model.eval()

    pred_raws = []
    sigmas = []
    qualities = []
    trues = []
    acts = []
    subjects = []

    for batch in loader:
        seq_ppg = batch["seq_ppg"].to(device, non_blocking=True)
        seq_acc = batch["seq_acc"].to(device, non_blocking=True)
        seq_valid = batch["seq_valid"].to(device, non_blocking=True)
        seq_act = batch["seq_act"].to(device, non_blocking=True)

        y = batch["hr"].to(device, non_blocking=True)

        out = model(seq_ppg, seq_acc, seq_valid, seq_act=seq_act)

        pred_raws.append(out["mu"].detach().cpu().numpy())
        sigmas.append(out["sigma"].detach().cpu().numpy())
        qualities.append(out["quality"].detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
        acts.append(batch["act"].detach().cpu().numpy())
        subjects.extend(list(batch["subject"]))

    pred_raw = np.concatenate(pred_raws).astype(np.float32) if pred_raws else np.asarray([], dtype=np.float32)
    sigma = np.concatenate(sigmas).astype(np.float32) if sigmas else np.asarray([], dtype=np.float32)
    quality = np.concatenate(qualities).astype(np.float32) if qualities else np.asarray([], dtype=np.float32)
    true = np.concatenate(trues).astype(np.float32) if trues else np.asarray([], dtype=np.float32)
    act = np.concatenate(acts).astype(np.int64) if acts else np.asarray([], dtype=np.int64)

    pred_raw = np.nan_to_num(pred_raw, nan=80.0, posinf=cfg.hr_max_bpm, neginf=cfg.hr_min_bpm)
    sigma = np.nan_to_num(sigma, nan=6.0, posinf=12.0, neginf=1.0)
    quality = np.nan_to_num(quality, nan=0.0, posinf=1.0, neginf=0.0)
    true = np.nan_to_num(true, nan=80.0, posinf=cfg.hr_max_bpm, neginf=cfg.hr_min_bpm)

    post = postprocess_predictions(pred_raw, sigma=sigma, quality=quality, cfg=cfg)
    pred_final = post["pred"]
    kept_mask = post["kept_mask"]

    mae_raw = float(mean_absolute_error(true, pred_raw)) if len(true) else float("nan")
    mae_final = float(mean_absolute_error(true, pred_final)) if len(true) else float("nan")

    if len(true) and kept_mask.any():
        retained_mae = float(mean_absolute_error(true[kept_mask], pred_final[kept_mask]))
        retained_ratio = float(np.mean(kept_mask))
    else:
        retained_mae = float("nan")
        retained_ratio = float("nan")

    return {
        "mae": mae_final,
        "mae_raw": mae_raw,
        "retained_mae": retained_mae,
        "retained_ratio": retained_ratio,
        "pred": pred_final,
        "pred_raw": pred_raw,
        "true": true,
        "act": act,
        "sigma": sigma,
        "quality": quality,
        "flags": post["flags"],
        "subject": np.asarray(subjects, dtype=object),
    }


def evaluate_validation_sets(
    model: HeartRateEstimator,
    loaders,
    device: torch.device,
    cfg: CFG,
) -> Dict[str, float]:
    maes = []
    raw_maes = []

    for loader in loaders:
        info = evaluate_loader(model, loader, device, cfg)

        if np.isfinite(info["mae"]):
            maes.append(float(info["mae"]))

        if np.isfinite(info["mae_raw"]):
            raw_maes.append(float(info["mae_raw"]))

    return {
        "final": float(np.mean(maes)) if maes else float("nan"),
        "raw": float(np.mean(raw_maes)) if raw_maes else float("nan"),
        "worst": float(np.max(maes)) if maes else float("nan"),
    }
