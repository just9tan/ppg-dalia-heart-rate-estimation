# PPG-Dalia Heart Rate Estimation

Research implementation for heart-rate estimation from wrist photoplethysmography (PPG) and accelerometer (ACC) signals using the PPG-Dalia dataset under strict leave-one-subject-out validation.

This repository provides a cleaned public implementation intended for academic, research, and reproducibility purposes. The dataset, private checkpoints, unpublished internal experiment settings, and manuscript-specific materials are not included.

---

## Overview

Wearable PPG-based heart-rate estimation is challenging because wrist signals are strongly affected by motion artifacts, especially during dynamic activities such as walking, cycling, stairs, and table soccer.

This project implements a deep learning pipeline that processes synchronized PPG and ACC windows, extracts temporal signal representations, estimates heart rate, and evaluates subject-level generalization using strict leave-one-subject-out validation.

The public pipeline includes:

- PPG-Dalia subject-level data loading
- Fixed-length PPG and ACC window extraction
- ACC resampling and signal alignment
- PPG and ACC normalization
- Artifact-aware preprocessing
- Temporal neural network modeling
- Uncertainty-aware prediction
- Post-processing and quality gating
- Subject-wise and activity-wise evaluation
- TorchScript model export support

---

## Repository Structure

```text
ppg-dalia-heart-rate-estimation/
├── configs/
│   └── ppg_dalia_loso.yaml
├── scripts/
│   ├── run_loso.py
│   ├── export_model.py
│   └── summarize_results.py
├── src/
│   ├── config.py
│   ├── signal_utils.py
│   ├── data.py
│   ├── network.py
│   ├── objectives.py
│   ├── inference.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── trainer.py
├── LICENSE
├── NOTICE
├── requirements.txt
└── README.md
