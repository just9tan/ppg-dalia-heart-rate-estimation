from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import CFG


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sigma = sigma.clamp_min(1e-3)
    z = (mu - target) / sigma

    return 0.5 * (z * z + 2.0 * torch.log(sigma))


def robust_tail_loss(errors: torch.Tensor, ratio: float = 0.20) -> torch.Tensor:
    errors = errors.reshape(-1)

    if errors.numel() == 0:
        return errors.new_tensor(0.0)

    k = max(1, int(round(float(ratio) * float(errors.numel()))))
    k = min(k, errors.numel())

    return torch.topk(errors, k=k, largest=True).values.mean()


def training_objective(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    cfg: CFG,
    act: torch.Tensor = None,
    quality: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mu = outputs["mu"]
    sigma = outputs["sigma"]

    nll = gaussian_nll(mu, sigma, target)
    huber = F.smooth_l1_loss(mu, target, beta=cfg.label_smooth_l1_beta, reduction="none")

    sample_w = torch.ones_like(target)

    if act is not None:
        dynamic = ((act == 2) | (act == 3) | (act == 4) | (act == 7)).float()
        sample_w = sample_w + 0.10 * dynamic

    if quality is not None:
        q = quality.float()
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        sample_w = sample_w + 0.05 * q.clamp(0.0, 5.0)

    high_hr = (target >= 130.0).float()
    sample_w = sample_w + 0.05 * high_hr
    sample_w = sample_w.clamp(1.0, 2.5)

    base_loss = torch.sum((nll + 0.20 * huber) * sample_w) / sample_w.sum().clamp_min(1e-6)
    tail_loss = robust_tail_loss(huber.detach() + torch.abs(mu - target), ratio=0.20)

    loss = base_loss + 0.05 * tail_loss

    stats = {
        "loss": float(loss.detach().cpu().item()),
        "nll": float(nll.mean().detach().cpu().item()),
        "huber": float(huber.mean().detach().cpu().item()),
        "tail": float(tail_loss.detach().cpu().item()),
    }

    return loss, stats
