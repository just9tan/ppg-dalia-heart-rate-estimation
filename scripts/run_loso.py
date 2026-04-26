import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CFG
from src.trainer import run_strict_loso


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run strict leave-one-subject-out evaluation on PPG-Dalia."
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val_k", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = CFG()
    cfg.data_dir = args.data_dir

    if args.epochs is not None:
        cfg.epochs = int(args.epochs)

    if args.batch is not None:
        cfg.batch = int(args.batch)

    if args.lr is not None:
        cfg.lr = float(args.lr)

    if args.val_k is not None:
        cfg.val_k = int(args.val_k)

    if args.no_plots:
        cfg.save_plots = False

    if not os.path.isdir(cfg.data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {cfg.data_dir}")

    run_strict_loso(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
