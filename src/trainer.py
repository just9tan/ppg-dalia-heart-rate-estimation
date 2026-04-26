import os
import copy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .config import CFG, ACTIVITY_NAMES
from .data import load_subject_datasets
from .network import HeartRateEstimator, InferenceWrapper, count_trainable_params
from .objectives import training_objective
from .evaluation import evaluate_loader, evaluate_validation_sets
from .visualization import plot_hr_timeline
from .signal_utils import seed_all


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()

        for key, value in self.shadow.items():
            if key not in model_state:
                continue

            src = model_state[key].detach()

            if torch.is_floating_point(value) and torch.is_floating_point(src):
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def build_loader(
    dataset: Dataset,
    cfg: CFG,
    device: torch.device,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_mem and device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )


def augment_batch(
    seq_ppg: torch.Tensor,
    seq_acc: torch.Tensor,
    cfg: CFG,
    training: bool = True,
) -> tuple:
    if not training:
        return seq_ppg, seq_acc

    seq_ppg = torch.nan_to_num(seq_ppg, nan=0.0, posinf=0.0, neginf=0.0)
    seq_acc = torch.nan_to_num(seq_acc, nan=0.0, posinf=0.0, neginf=0.0)

    if cfg.ppg_noise_std > 0:
        seq_ppg = seq_ppg + cfg.ppg_noise_std * torch.randn_like(seq_ppg)

    if cfg.acc_noise_std > 0:
        seq_acc = seq_acc + cfg.acc_noise_std * torch.randn_like(seq_acc)

    if cfg.amp_scale_std > 0:
        scale = 1.0 + cfg.amp_scale_std * torch.randn(
            seq_ppg.size(0),
            1,
            1,
            1,
            device=seq_ppg.device,
            dtype=seq_ppg.dtype,
        )

        seq_ppg = seq_ppg * scale

    return seq_ppg, seq_acc


def train_fold(
    train_ds: Dataset,
    val_sets: List[Dataset],
    cfg: CFG,
    device: torch.device,
) -> HeartRateEstimator:
    train_loader = build_loader(train_ds, cfg, device, shuffle=True)
    val_loaders = [build_loader(vs, cfg, device, shuffle=False) for vs in val_sets]

    model = HeartRateEstimator(cfg).to(device)
    ema = EMA(model, decay=cfg.ema_decay)
    ema_model = copy.deepcopy(model).to(device).eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=cfg.min_lr,
    )

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(cfg.amp and device.type == "cuda"),
        )

        def autocast():
            return torch.amp.autocast(
                device_type="cuda",
                enabled=(cfg.amp and device.type == "cuda"),
            )
    else:
        scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.amp and device.type == "cuda"),
        )

        def autocast():
            return torch.cuda.amp.autocast(
                enabled=(cfg.amp and device.type == "cuda"),
            )

    best_score = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        total_loss = 0.0
        n_seen = 0

        for batch in train_loader:
            seq_ppg = batch["seq_ppg"].to(device, non_blocking=True)
            seq_acc = batch["seq_acc"].to(device, non_blocking=True)
            seq_valid = batch["seq_valid"].to(device, non_blocking=True)
            seq_act = batch["seq_act"].to(device, non_blocking=True)
            y = batch["hr"].to(device, non_blocking=True)
            act = batch["act"].to(device, non_blocking=True)
            quality = batch["quality"].to(device, non_blocking=True)

            seq_ppg, seq_acc = augment_batch(seq_ppg, seq_acc, cfg, training=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(seq_ppg, seq_acc, seq_valid, seq_act=seq_act)
                loss, stats = training_objective(
                    outputs,
                    y,
                    cfg,
                    act=act,
                    quality=quality,
                )

            scaler.scale(loss).backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            bs = seq_ppg.size(0)
            total_loss += float(loss.detach().cpu().item()) * bs
            n_seen += bs

        ema.copy_to(ema_model)

        val_scores = evaluate_validation_sets(ema_model, val_loaders, device, cfg)
        val_score = float(val_scores["final"])

        scheduler.step(val_score)

        train_loss = total_loss / max(1, n_seen)
        lr_now = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:03d} | "
            f"lr={lr_now:.2e} | "
            f"loss={train_loss:.4f} | "
            f"val_raw={val_scores['raw']:.3f} | "
            f"val_final={val_scores['final']:.3f} | "
            f"val_worst={val_scores['worst']:.3f}"
        )

        if np.isfinite(val_score) and val_score < best_score - 1e-4:
            best_score = val_score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in ema_model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

            if no_improve >= cfg.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        ema_model.load_state_dict(best_state)

    return ema_model


def run_strict_loso(cfg: CFG, seed: int = 42) -> None:
    seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    subject_datasets = load_subject_datasets(cfg)
    available = sorted(subject_datasets.keys(), key=lambda x: int(x[1:]))

    if len(available) < 3:
        raise RuntimeError("Need at least 3 available subjects for strict LOSO.")

    os.makedirs(cfg.plot_dir, exist_ok=True)

    results = []
    per_activity = []

    for fold_idx, test_subject in enumerate(available, start=1):
        remaining = [s for s in available if s != test_subject]

        rng = np.random.default_rng(seed + fold_idx)
        remaining = list(rng.permutation(remaining))

        val_k = max(1, min(int(cfg.val_k), len(remaining) - 1))
        val_subjects = remaining[:val_k]
        train_subjects = remaining[val_k:]

        print("=" * 80)
        print(
            f"Fold {fold_idx}/{len(available)} | "
            f"test={test_subject} | "
            f"val={val_subjects} | "
            f"train={train_subjects}"
        )

        train_ds = ConcatDataset([subject_datasets[s] for s in train_subjects])
        val_sets = [subject_datasets[s] for s in val_subjects]
        test_ds = subject_datasets[test_subject]

        test_loader = build_loader(test_ds, cfg, device, shuffle=False)

        model = train_fold(train_ds, val_sets, cfg, device)

        if fold_idx == 1:
            print(f"Trainable parameters: {count_trainable_params(model):,}")

        test_info = evaluate_loader(model, test_loader, device, cfg)

        print(
            f"Test {test_subject}: "
            f"raw MAE={test_info['mae_raw']:.3f} bpm | "
            f"final MAE={test_info['mae']:.3f} bpm | "
            f"N={len(test_info['true'])}"
        )

        results.append(
            {
                "subject": test_subject,
                "mae_raw": float(test_info["mae_raw"]),
                "mae": float(test_info["mae"]),
                "retained_mae": float(test_info["retained_mae"]),
                "retained_ratio": float(test_info["retained_ratio"]),
                "n": int(len(test_info["true"])),
                "val_subjects": ",".join(val_subjects),
            }
        )

        true = test_info["true"]
        pred = test_info["pred"]
        act = test_info["act"]

        for activity_id in np.unique(act):
            mask = act == activity_id

            if int(mask.sum()) < 5:
                continue

            per_activity.append(
                {
                    "subject": test_subject,
                    "activity_id": int(activity_id),
                    "activity_name": ACTIVITY_NAMES.get(int(activity_id), str(activity_id)),
                    "mae": float(mean_absolute_error(true[mask], pred[mask])),
                    "n": int(mask.sum()),
                }
            )

        if cfg.save_plots:
            plot_hr_timeline(
                test_subject,
                true,
                pred,
                act,
                save_dir=cfg.plot_dir,
                step_sec=cfg.step_sec,
                suffix="final",
            )

    df = pd.DataFrame(results).sort_values("subject")
    df.to_csv(cfg.result_csv, index=False)

    print(f"Saved {cfg.result_csv}")
    print(df)

    if per_activity:
        df_act = pd.DataFrame(per_activity).sort_values(["subject", "activity_id"])
        df_act.to_csv(cfg.result_activity_csv, index=False)
        print(f"Saved {cfg.result_activity_csv}")

    print()
    print(f"Overall raw MAE   : {df['mae_raw'].mean():.3f} bpm")
    print(f"Overall final MAE : {df['mae'].mean():.3f} ± {df['mae'].std(ddof=1):.3f} bpm")


def train_and_export(
    cfg: CFG,
    seed: int = 42,
    export_path: str = "checkpoints/ppg_dalia_hr_estimator.pt",
    val_subjects: Optional[List[str]] = None,
) -> str:
    seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    subject_datasets = load_subject_datasets(cfg)
    available = sorted(subject_datasets.keys(), key=lambda x: int(x[1:]))

    if len(available) < 3:
        raise RuntimeError("Need at least 3 available subjects.")

    if val_subjects is None:
        val_k = max(1, min(int(cfg.val_k), len(available) - 1))
        val_subjects = available[-val_k:]

    train_subjects = [s for s in available if s not in val_subjects]

    train_ds = ConcatDataset([subject_datasets[s] for s in train_subjects])
    val_sets = [subject_datasets[s] for s in val_subjects]

    print(f"Final training | train={train_subjects} | val={val_subjects}")

    model = train_fold(train_ds, val_sets, cfg, device).to("cpu").eval()
    wrapper = InferenceWrapper(model).eval()

    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    scripted = torch.jit.script(wrapper)
    scripted = torch.jit.freeze(scripted)
    torch.jit.save(scripted, export_path)

    print(f"Exported TorchScript model: {export_path}")

    return export_path
