from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        hidden = max(4, channels // reduction)

        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)

        return x * s


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()

        padding = (kernel_size // 2) * dilation

        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )

        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.ff = nn.Conv1d(channels, channels, kernel_size=1)
        self.se = SEBlock1D(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        y = self.ff(y)
        y = self.norm2(y)
        y = self.se(y)

        return F.gelu(x + y)


class SignalEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, dropout: float):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )

        self.block1 = DepthwiseResidualBlock(base_channels, dilation=1, dropout=dropout)

        self.down1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU(),
        )

        self.block2 = DepthwiseResidualBlock(base_channels * 2, dilation=2, dropout=dropout)

        self.down2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU(),
        )

        self.block3 = DepthwiseResidualBlock(base_channels * 2, dilation=4, dropout=dropout)

        self.out_channels = base_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)

        return x


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, heads: int, layers: int, ff_mult: int, dropout: float, max_len: int):
        super().__init__()

        self.proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.pos_emb[:, : x.size(1), :]

        pad_mask = valid_mask <= 0.5

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        return x


class FrequencyFeatures(nn.Module):
    def __init__(self, cfg: CFG, out_dim: int = 16):
        super().__init__()

        self.cfg = cfg
        self.win_len = int(cfg.win_sec * cfg.bvp_fs)

        self.register_buffer("hann", torch.hann_window(self.win_len), persistent=False)

        self.net = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, ppg: torch.Tensor):
        x = ppg.squeeze(1)

        if x.size(1) != self.win_len:
            x = F.interpolate(x.unsqueeze(1), size=self.win_len, mode="linear", align_corners=False).squeeze(1)

        x = x * self.hann[None, :]

        spec = torch.fft.rfft(x, dim=1)
        mag = torch.abs(spec)

        freqs = torch.fft.rfftfreq(self.win_len, d=1.0 / float(self.cfg.bvp_fs)).to(ppg.device)
        bpm = freqs * 60.0

        mask = (bpm >= self.cfg.hr_min_bpm) & (bpm <= self.cfg.hr_max_bpm)

        band = mag[:, mask]
        bpm_band = bpm[mask]

        idx = torch.argmax(band, dim=1)
        hr0 = bpm_band[idx].clamp(self.cfg.hr_min_bpm, self.cfg.hr_max_bpm)

        peak = band.gather(1, idx[:, None]).squeeze(1)
        median = torch.median(band, dim=1).values + 1e-6
        ratio = (peak / median).clamp(1.0, 30.0)

        entropy = band / (band.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(entropy * torch.log(entropy + 1e-8)).sum(dim=1)

        feat = torch.stack(
            [
                (hr0 - 90.0) / 40.0,
                torch.log(ratio + 1e-6) / 3.0,
                entropy / 5.0,
            ],
            dim=1,
        )

        feat = self.net(feat)

        return hr0, ratio, feat


class HeartRateEstimator(nn.Module):
    """
    Public clean model.

    This model keeps the repository useful and reproducible at a high level,
    while avoiding publication of private experimental tuning details.
    """

    def __init__(self, cfg: CFG):
        super().__init__()

        self.cfg = cfg

        self.ppg_encoder = SignalEncoder(
            in_channels=1,
            base_channels=cfg.base_ch,
            dropout=cfg.dropout,
        )

        self.acc_encoder = SignalEncoder(
            in_channels=3,
            base_channels=max(8, cfg.base_ch // 2),
            dropout=cfg.dropout,
        )

        self.freq = FrequencyFeatures(cfg, out_dim=16)

        self.act_emb = nn.Embedding(9, cfg.act_emb_dim)

        ppg_dim = self.ppg_encoder.out_channels
        acc_dim = self.acc_encoder.out_channels
        token_dim = ppg_dim + acc_dim + 16 + cfg.act_emb_dim + 3

        self.temporal = TemporalEncoder(
            in_dim=token_dim,
            d_model=cfg.seq_d_model,
            heads=cfg.seq_heads,
            layers=cfg.seq_layers,
            ff_mult=cfg.seq_ff_mult,
            dropout=cfg.seq_dropout,
            max_len=cfg.seq_len,
        )

        head_dim = cfg.seq_d_model * 2 + 16 + cfg.act_emb_dim + 3

        self.head = nn.Sequential(
            nn.Linear(head_dim, 192),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(cfg.dropout * 0.5),
        )

        self.hr_head = nn.Linear(96, 1)
        self.sigma_head = nn.Linear(96, 1)
        self.quality_head = nn.Linear(96, 1)

    def forward(
        self,
        seq_ppg: torch.Tensor,
        seq_acc: torch.Tensor,
        seq_valid: torch.Tensor,
        seq_act: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        seq_ppg = torch.nan_to_num(seq_ppg, nan=0.0, posinf=0.0, neginf=0.0)
        seq_acc = torch.nan_to_num(seq_acc, nan=0.0, posinf=0.0, neginf=0.0)
        seq_valid = torch.nan_to_num(seq_valid, nan=0.0, posinf=0.0, neginf=0.0).float()

        B, S, _, T = seq_ppg.shape

        if seq_act is None:
            seq_act = torch.zeros(B, S, dtype=torch.long, device=seq_ppg.device)
        else:
            seq_act = seq_act.to(seq_ppg.device).long().clamp(0, 8)

        flat_ppg = seq_ppg.reshape(B * S, 1, T)
        flat_acc = seq_acc.reshape(B * S, 3, T)

        ppg_feat = self.ppg_encoder(flat_ppg)
        acc_feat = self.acc_encoder(flat_acc)

        ppg_pool = F.adaptive_avg_pool1d(ppg_feat, 1).squeeze(-1).reshape(B, S, -1)
        acc_pool = F.adaptive_avg_pool1d(acc_feat, 1).squeeze(-1).reshape(B, S, -1)

        hr0_flat, peak_ratio_flat, freq_feat_flat = self.freq(flat_ppg)
        hr0 = hr0_flat.reshape(B, S)
        peak_ratio = peak_ratio_flat.reshape(B, S)
        freq_feat = freq_feat_flat.reshape(B, S, -1)

        act_feat = self.act_emb(seq_act)

        hr0_prev = torch.cat([hr0[:, :1], hr0[:, :-1]], dim=1)

        extras = torch.stack(
            [
                (hr0 - 90.0) / 40.0,
                torch.log(peak_ratio + 1e-6) / 3.0,
                (hr0 - hr0_prev).clamp(-25.0, 25.0) / 25.0,
            ],
            dim=-1,
        )

        tokens = torch.cat([ppg_pool, acc_pool, freq_feat, act_feat, extras], dim=-1)

        enc = self.temporal(tokens, seq_valid)

        valid = seq_valid.unsqueeze(-1)
        enc_mean = (enc * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        enc_last = enc[:, -1, :]

        cur_freq = freq_feat[:, -1, :]
        cur_act = act_feat[:, -1, :]
        cur_extra = extras[:, -1, :]

        h = torch.cat([enc_last, enc_mean, cur_freq, cur_act, cur_extra], dim=1)
        h = self.head(h)

        delta = 28.0 * torch.tanh(self.hr_head(h).squeeze(-1))
        mu = (hr0[:, -1] + delta).clamp(self.cfg.hr_min_bpm, self.cfg.hr_max_bpm)

        sigma = 1.0 + 10.0 * torch.sigmoid(self.sigma_head(h).squeeze(-1))
        quality = torch.sigmoid(self.quality_head(h).squeeze(-1))

        return {
            "mu": mu,
            "sigma": sigma,
            "quality": quality,
            "hr0": hr0[:, -1],
            "peak_ratio": peak_ratio[:, -1],
        }


class InferenceWrapper(nn.Module):
    def __init__(self, model: HeartRateEstimator):
        super().__init__()
        self.model = model

    def forward(
        self,
        seq_ppg: torch.Tensor,
        seq_acc: torch.Tensor,
        seq_valid: torch.Tensor,
        seq_act: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.model(seq_ppg, seq_acc, seq_valid, seq_act=seq_act)

        return out["mu"]


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
