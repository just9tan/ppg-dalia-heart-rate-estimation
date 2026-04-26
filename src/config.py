import os
from dataclasses import dataclass


@dataclass
class CFG:
    # Dataset path should be provided from CLI.
    # Do not upload the PPG-Dalia dataset to GitHub.
    data_dir: str = "data/PPG_FieldStudy"

    # Dataset
    bvp_fs: int = 64
    acc_fs: int = 32
    win_sec: int = 8
    step_sec: int = 2
    act_fs: int = 4
    hr_min_bpm: float = 40.0
    hr_max_bpm: float = 200.0

    # Training
    batch: int = 128
    epochs: int = 60
    lr: float = 3.0e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    patience: int = 10
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.995
    label_smooth_l1_beta: float = 4.0

    # Backbone / temporal model
    base_ch: int = 32
    attn_dim: int = 96
    attn_heads: int = 4
    token_len: int = 32
    dropout: float = 0.20
    ppg_only_prob: float = 0.08
    act_emb_dim: int = 8
    severe_emb_dim: int = 4

    # Sequence modeling
    seq_len: int = 10
    seq_d_model: int = 128
    seq_heads: int = 4
    seq_layers: int = 3
    seq_ff_mult: int = 4
    seq_dropout: float = 0.18

    # Optional pretrained encoder interface
    pretrained_encoder_path: str = ""
    pretrained_encoder_format: str = "auto"
    pretrained_encoder_strict: bool = False
    freeze_pretrained_epochs: int = 0
    encoder_proj_dim: int = 0

    # Posterior / temporal decoding
    posterior_bin_step: float = 2.0
    posterior_target_sigma_bpm: float = 3.0
    decode_enable: bool = True
    decode_blend_alpha: float = 0.70
    decode_obs_mix: float = 0.40

    decode_transition_rest_bpm: float = 5.5
    decode_transition_work_bpm: float = 6.5
    decode_transition_drive_bpm: float = 8.0
    decode_transition_dyn_bpm: float = 11.0

    adaptive_decode_enable: bool = True
    decode_alpha_easy: float = 0.20
    decode_alpha_mid: float = 0.50
    decode_alpha_hard: float = 0.80
    decode_easy_sigma_thr: float = 4.5
    decode_hard_sigma_thr: float = 7.0
    decode_easy_top1_thr: float = 0.55
    decode_hard_top1_thr: float = 0.36
    decode_easy_entropy_thr: float = 0.38
    decode_hard_entropy_thr: float = 0.62
    decode_easy_artifact_thr: float = 0.25
    decode_hard_artifact_thr: float = 0.55
    raw_lock_keep_easy: bool = True
    decode_alpha_smooth_k: int = 5

    # Raw trust logic
    raw_trust_enable: bool = True
    raw_trust_sigma_max: float = 5.8
    raw_trust_artifact_max: float = 0.45
    raw_trust_top1_min: float = 0.34
    raw_trust_entropy_max: float = 0.72
    raw_trust_jump_bpm: float = 10.0
    raw_trust_reg_diff_bpm: float = 7.0
    raw_trust_post_diff_bpm: float = 8.0
    raw_trust_local_diff_bpm: float = 6.5
    raw_trust_min_checks: int = 6
    raw_trust_smooth_k: int = 5
    decode_oppose_post_diff_bpm: float = 12.0
    decode_oppose_top1_min: float = 0.52
    decode_oppose_entropy_max: float = 0.62
    decode_oppose_margin_bpm: float = 6.0

    # Regime-conditioned decoding
    regime_transition_static_bpm: float = 4.5
    regime_transition_loco_bpm: float = 8.0
    regime_transition_ballistic_bpm: float = 12.0
    regime_ballistic_artifact_gamma: float = 1.0
    regime_transition_severe_gamma: float = 0.18
    belief_obs_sigma_floor: float = 2.5

    # Loss weights
    loss_ce_weight: float = 0.42
    loss_artifact_bce_weight: float = 0.08
    loss_consistency_weight: float = 0.05
    loss_delta_weight: float = 0.12
    delta_label_smooth_l1_beta: float = 3.0
    delta_clip_bpm: float = 25.0

    upward_transition_weight_gamma: float = 0.12
    upward_transition_thr: float = 6.0
    severe_weight_gamma: float = 0.30
    dynamic_weight_gamma: float = 0.10
    high_hr_weight_gamma: float = 0.05
    high_hr_weight_thr: float = 145.0

    loss_peak_under_weight: float = 0.18
    loss_peak_prior_weight: float = 0.05
    peak_loss_hr_thr: float = 130.0
    peak_under_margin_bpm: float = 2.0
    peak_prior_ratio_thr: float = 2.20
    peak_prior_margin_bpm: float = 8.0

    loss_asym_high_hr_weight: float = 0.16
    asym_high_hr_thr: float = 120.0
    asym_high_hr_margin_bpm: float = 1.5
    asym_high_hr_gamma: float = 1.50

    loss_regime_ce_weight: float = 0.08
    loss_tail_cvar_weight: float = 0.18
    cvar_alpha: float = 0.20
    topk_tail_ratio: float = 0.18

    # Augmentation
    aug_warmup_epochs: int = 8
    ppg_noise_std: float = 0.010
    acc_noise_std: float = 0.006
    amp_scale_std: float = 0.06
    time_mask_prob: float = 0.10
    time_mask_width: int = 24

    # Adaptive filtering
    use_offline_adaptive: bool = True
    adaptive_lags: int = 3
    adaptive_ridge: float = 1e-2
    adaptive_use_derivative: bool = True
    adaptive_activity_chunk_norm: bool = True
    adaptive_norm_clip: float = 5.0

    # Validation / sampling
    val_k: int = 3
    hard_sampler_warmup_epochs: int = 4
    use_hard_sampler: bool = True
    train_eval_subset: int = 0

    sampler_subject_gamma: float = 1.35
    sampler_activity_gamma: float = 0.50
    sampler_hrbin_gamma: float = 0.30
    sampler_dynamic_boost: float = 0.15
    sampler_highhr_boost: float = 0.18
    sampler_ballistic_boost: float = 0.18

    # Post-processing
    medfilt_k: int = 5
    report_retained_mae: bool = True
    retained_sigma_max: float = 8.0
    gate_mask_only: bool = False
    gate_uncertain_alpha: float = 0.10

    sqi_t_low: float = 0.42
    sqi_t_high: float = 0.60
    sqi_sigma_soft: float = 7.0
    sqi_sigma_hard: float = 9.0
    sqi_jump_soft: float = 10.0
    sqi_jump_hard: float = 18.0

    gate_up_margin_bpm: float = 3.0
    gate_up_alpha: float = 0.86
    gate_dynamic_up_alpha: float = 0.90
    gate_up_alpha_severe: float = 0.74
    gate_dynamic_up_alpha_severe: float = 0.82

    # Severe motion block detection
    severe_gate_enable: bool = True
    severe_scan_step_sec: float = 0.5
    severe_merge_gap_sec: float = 1.0
    severe_ppg_burst_thr: float = 0.30
    severe_acc_burst_thr: float = 0.20
    severe_ppg_peak_thr: float = 10.0
    severe_acc_peak_thr: float = 8.0
    severe_spike_z_thr: float = 3.0
    severe_norm_clip: float = 12.0
    severe_light_min_sec: float = 8.0
    severe_blend_min_sec: float = 10.0
    severe_reject_min_sec: float = 18.0
    severe_light_alpha: float = 0.50
    severe_blend_alpha: float = 0.20
    severe_reject_alpha: float = 0.00

    # Rescue guards
    dip_guard_enable: bool = True
    dip_guard_window: int = 7
    dip_guard_drop_bpm: float = 12.0
    dip_guard_dynamic_drop_bpm: float = 10.0
    dip_guard_support_margin: float = 6.0
    dip_guard_sigma_min: float = 5.5
    dip_guard_artifact_min: float = 0.20
    dip_guard_min_votes: int = 3
    dip_guard_lift_alpha: float = 0.60
    dip_guard_lift_alpha_dynamic: float = 0.70

    peak_guard_enable: bool = True
    peak_guard_peakratio_min: float = 2.25
    peak_guard_hr0_margin_bpm: float = 10.0
    peak_guard_sigma_max: float = 9.5
    peak_guard_artifact_max: float = 0.75
    peak_guard_alpha: float = 0.52
    peak_guard_alpha_dynamic: float = 0.68

    # Validation score
    score_final_weight: float = 0.40
    score_topk_tail_weight: float = 0.35
    score_worst_val_weight: float = 0.25
    score_tail_topk: int = 4

    # Export
    export_dynamic_int8: bool = False

    # Outputs
    save_plots: bool = True
    plot_dir: str = "plots_ppg_dalia_loso"
    result_csv: str = "strict_loso_results.csv"
    result_activity_csv: str = "strict_loso_results_by_activity.csv"

    # Dataloader
    num_workers: int = 0 if os.name == "nt" else 4
    pin_mem: bool = True


ACTIVITY_NAMES = {
    0: "Transient",
    1: "Sitting",
    2: "Stairs",
    3: "Table Soccer",
    4: "Cycling",
    5: "Driving",
    6: "Lunch",
    7: "Walking",
    8: "Working",
}
