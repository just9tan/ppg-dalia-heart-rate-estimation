import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CFG
from src.trainer import train_and_export


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train final model and export a TorchScript checkpoint."
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--export_path",
        type=str,
        default="checkpoints/ppg_dalia_hr_estimator.pt",
    )
    parser.add_argument("--val_subjects", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = CFG()
    cfg.data_dir = args.data_dir

    if args.epochs is not None:
        cfg.epochs = int(args.epochs)

    if args.batch is not None:
        cfg.batch = int(args.batch)

    if not os.path.isdir(cfg.data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {cfg.data_dir}")

    val_subjects = None

    if args.val_subjects:
        val_subjects = [s.strip() for s in args.val_subjects.split(",")]

    train_and_export(
        cfg,
        seed=args.seed,
        export_path=args.export_path,
        val_subjects=val_subjects,
    )


if __name__ == "__main__":
    main()
