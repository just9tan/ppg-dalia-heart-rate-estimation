import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Summarize subject-wise strict LOSO results."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="strict_loso_results.csv",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    print("Subject-wise results")
    print(df)

    print()
    print("Summary")

    if "mae_raw" in df.columns:
        print(f"Mean raw MAE   : {df['mae_raw'].mean():.3f} bpm")

    if "mae" in df.columns:
        print(f"Mean final MAE : {df['mae'].mean():.3f} bpm")
        print(f"STD final MAE  : {df['mae'].std(ddof=1):.3f} bpm")

    if "retained_mae" in df.columns:
        print(f"Mean kept MAE  : {df['retained_mae'].mean():.3f} bpm")

    if "retained_ratio" in df.columns:
        print(f"Retained ratio : {100 * df['retained_ratio'].mean():.2f}%")


if __name__ == "__main__":
    main()
