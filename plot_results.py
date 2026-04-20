import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot experiment logs")
    parser.add_argument("--logs", nargs="+", required=True, help="CSV log files")
    parser.add_argument("--output_dir", type=str, default="outputs/plots")
    parser.add_argument(
        "--mode",
        type=str,
        default="model_compare",
        choices=["model_compare", "backbone_compare"],
    )
    return parser.parse_args()


def load_logs(log_paths):
    frames = []
    for log_path in log_paths:
        path = Path(log_path)
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_output_stem(log_paths):
    stems = [Path(log_path).stem for log_path in log_paths]
    if len(stems) == 1:
        return stems[0]
    preview = "__".join(stems[:3])
    if len(stems) > 3:
        preview += f"__plus{len(stems) - 3}"
    return preview


def plot_curves(df, output_dir, mode, output_stem):
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "model_compare":
        group_key = "model"
        title_suffix = "Model Comparison"
    else:
        group_key = "backbone"
        title_suffix = "Backbone Comparison"

    plt.figure(figsize=(10, 6))
    for name, group in df.groupby(group_key):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group["train_loss"], marker="o", label=f"{name} train")
        plt.plot(group["epoch"], group["val_loss"], marker="s", linestyle="--", label=f"{name} val")

    dataset_names = ", ".join(sorted(df["dataset"].dropna().astype(str).unique()))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_suffix} ({dataset_names})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{output_stem}_loss_curve.png", dpi=200)
    plt.close()


def export_summary(df, output_dir, mode, output_stem):
    if mode == "model_compare":
        summary = (
            df.sort_values("epoch")
            .groupby(["model", "dataset"], as_index=False)
            .tail(1)
            .loc[:, ["model", "backbone", "dataset", "epoch", "train_loss", "val_loss", "lr"]]
            .sort_values(["dataset", "model"])
        )
    else:
        summary = (
            df.sort_values("epoch")
            .groupby(["model", "backbone", "dataset"], as_index=False)
            .tail(1)
            .loc[:, ["model", "backbone", "dataset", "epoch", "train_loss", "val_loss", "lr"]]
            .sort_values(["model", "dataset", "backbone"])
        )

    summary.to_csv(output_dir / f"{output_stem}_summary.csv", index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    df = load_logs(args.logs)
    output_stem = build_output_stem(args.logs)
    plot_curves(df, output_dir, args.mode, output_stem)
    export_summary(df, output_dir, args.mode, output_stem)


if __name__ == "__main__":
    main()
