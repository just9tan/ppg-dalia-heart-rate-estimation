import os

import numpy as np
import matplotlib.pyplot as plt


def plot_hr_timeline(
    subject_id: str,
    true_hr: np.ndarray,
    pred_hr: np.ndarray,
    acts: np.ndarray,
    save_dir: str,
    step_sec: int = 2,
    suffix: str = "final",
) -> None:
    true_hr = np.asarray(true_hr, dtype=float)
    pred_hr = np.asarray(pred_hr, dtype=float)
    acts = np.asarray(acts, dtype=int)

    if len(true_hr) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    t = np.arange(len(true_hr)) * step_sec

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t, true_hr, label="True HR")
    ax.plot(t, pred_hr, "--", label="Pred HR")

    ax.set_title(f"{subject_id} — Heart Rate over time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Heart Rate [bpm]")
    ax.set_ylim(30, 210)

    if len(acts):
        cur = int(acts[0])
        start = 0

        for i in range(1, len(acts)):
            if int(acts[i]) != cur:
                if cur != 0:
                    ax.axvspan(start * step_sec, i * step_sec, color=f"C{cur % 10}", alpha=0.10)

                start = i
                cur = int(acts[i])

        if cur != 0:
            ax.axvspan(start * step_sec, len(acts) * step_sec, color=f"C{cur % 10}", alpha=0.10)

    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"{subject_id}_timeline_{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
