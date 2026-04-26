import os
import pickle
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CFG
from .signal_utils import (
    nan_to_num_np,
    robust_zscore_1d,
    robust_zscore_2d,
    extract_1d,
    extract_2d,
    resample_linear_np,
    ensure_three_axis_acc,
    filter_ppg_window,
    compute_window_quality,
)


class PPGSequenceDataset(Dataset):
    """
    Public dataset wrapper for PPG-Dalia.

    It loads one subject file, extracts synchronized PPG and ACC windows,
    and builds short temporal sequences for heart-rate estimation.
    """

    def __init__(self, pkl_path: str, subject_id: str, cfg: CFG):
        self.cfg = cfg
        self.subject_id = subject_id

        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"Subject file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        if "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        bvp = np.asarray(data["signal"]["wrist"]["BVP"]).reshape(-1).astype(np.float32)
        acc = np.asarray(data["signal"]["wrist"]["ACC"]).astype(np.float32)
        labels = np.asarray(data["label"]).reshape(-1).astype(np.float32)

        acc = ensure_three_axis_acc(acc)

        activity = data.get("activity", None)

        if activity is None:
            n_act = int(len(bvp) / cfg.bvp_fs * cfg.act_fs)
            activity = np.zeros(n_act, dtype=np.int16)
        else:
            activity = np.asarray(activity)

            if activity.ndim > 1:
                activity = np.squeeze(activity)

            activity = np.nan_to_num(activity, nan=0).astype(np.int16)

        valid_label_idx = np.where(
            (labels > cfg.hr_min_bpm) & (labels < cfg.hr_max_bpm)
        )[0]

        bvp_len = int(cfg.win_sec * cfg.bvp_fs)
        acc_len = int(cfg.win_sec * cfg.acc_fs)

        self.ppg: List[np.ndarray] = []
        self.acc: List[np.ndarray] = []
        self.hr: List[float] = []
        self.act: List[int] = []
        self.subj: List[str] = []
        self.quality: List[float] = []
        self.label_idx: List[int] = []

        for label_i in valid_label_idx:
            t_sec = float(label_i * cfg.step_sec)

            b0 = int(round(t_sec * cfg.bvp_fs))
            a0 = int(round(t_sec * cfg.acc_fs))

            act_idx = int(round(t_sec * cfg.act_fs))
            act_idx = min(max(act_idx, 0), len(activity) - 1) if len(activity) else 0
            act_id = int(activity[act_idx]) if len(activity) else 0
            act_id = max(0, min(act_id, 8))

            ppg_raw = extract_1d(bvp, b0, bvp_len).astype(np.float32)
            acc_raw = extract_2d(acc, a0, acc_len).astype(np.float32)
            acc_up = resample_linear_np(acc_raw, out_len=bvp_len).astype(np.float32)

            ppg_norm = robust_zscore_1d(ppg_raw)
            acc_norm = robust_zscore_2d(acc_up)

            if cfg.use_offline_adaptive:
                ppg_proc = filter_ppg_window(
                    ppg_norm,
                    acc_norm,
                    lags=cfg.adaptive_lags,
                    ridge=cfg.adaptive_ridge,
                    use_derivative=cfg.adaptive_use_derivative,
                )
            else:
                ppg_proc = ppg_norm

            _, acc_energy = compute_window_quality(ppg_proc, acc_norm)

            self.ppg.append(ppg_proc.astype(np.float32))
            self.acc.append(acc_norm.astype(np.float32))
            self.hr.append(float(labels[label_i]))
            self.act.append(act_id)
            self.subj.append(str(subject_id))
            self.quality.append(float(acc_energy))
            self.label_idx.append(int(label_i))

        if len(self.hr) == 0:
            raise RuntimeError(f"No valid HR labels found for subject {subject_id}")

        self.ppg = np.stack(self.ppg, axis=0).astype(np.float32)
        self.acc = np.stack(self.acc, axis=0).astype(np.float32)
        self.hr = np.asarray(self.hr, dtype=np.float32)
        self.act = np.asarray(self.act, dtype=np.int64)
        self.subj = np.asarray(self.subj, dtype=object)
        self.quality = np.asarray(self.quality, dtype=np.float32)
        self.label_idx = np.asarray(self.label_idx, dtype=np.int64)

        self.seq_indices = self._build_sequence_indices()

    def _build_sequence_indices(self) -> np.ndarray:
        n = len(self.hr)
        seq_len = int(self.cfg.seq_len)

        seq_indices = np.zeros((n, seq_len), dtype=np.int64)

        for i in range(n):
            start = max(0, i - seq_len + 1)
            hist = list(range(start, i + 1))

            if len(hist) < seq_len:
                hist = [hist[0]] * (seq_len - len(hist)) + hist

            seq_indices[i] = np.asarray(hist[-seq_len:], dtype=np.int64)

        return seq_indices

    def __len__(self) -> int:
        return int(len(self.hr))

    def __getitem__(self, idx: int):
        seq_idx = self.seq_indices[idx]

        seq_ppg = self.ppg[seq_idx]                  # S, T
        seq_acc = self.acc[seq_idx]                  # S, T, 3
        seq_act = self.act[seq_idx]                  # S

        seq_ppg = torch.from_numpy(nan_to_num_np(seq_ppg)).unsqueeze(1).float()
        seq_acc = torch.from_numpy(nan_to_num_np(seq_acc)).permute(0, 2, 1).contiguous().float()
        seq_act = torch.from_numpy(seq_act.astype(np.int64)).long()

        y = torch.tensor(float(self.hr[idx]), dtype=torch.float32)
        act = torch.tensor(int(self.act[idx]), dtype=torch.long)
        quality = torch.tensor(float(self.quality[idx]), dtype=torch.float32)

        seq_valid = torch.ones(int(self.cfg.seq_len), dtype=torch.float32)

        return {
            "seq_ppg": seq_ppg,
            "seq_acc": seq_acc,
            "seq_act": seq_act,
            "seq_valid": seq_valid,
            "hr": y,
            "act": act,
            "quality": quality,
            "subject": str(self.subj[idx]),
        }


def load_subject_datasets(cfg: CFG) -> Dict[str, PPGSequenceDataset]:
    subjects = [f"S{i}" for i in range(1, 16)]
    out: Dict[str, PPGSequenceDataset] = {}

    for subject_id in subjects:
        pkl_path = os.path.join(cfg.data_dir, subject_id, f"{subject_id}.pkl")

        if not os.path.isfile(pkl_path):
            continue

        try:
            ds = PPGSequenceDataset(pkl_path, subject_id=subject_id, cfg=cfg)

            if len(ds) > 0:
                out[subject_id] = ds
                print(f"Loaded {subject_id}: {len(ds)} samples")

        except Exception as exc:
            print(f"Skipped {subject_id}: {exc}")

    return out
