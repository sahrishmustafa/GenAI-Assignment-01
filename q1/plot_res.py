import pandas as pd
import matplotlib.pyplot as plt

def plot_ablation_results(csv_path, out_dir="ablation_charts"):
    # Load results
    df = pd.read_csv(csv_path)

    # Ensure output dir exists
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Group experiments
    groups = {
        "Learning Rate": df[df["exp"].str.startswith("lr_")],
        "Batch Size": df[df["exp"].str.startswith("bs_")],
        "Filters": df[df["exp"].str.startswith("filters_")],
        "Layers": df[df["exp"].str.startswith("layers_")],
    }

    metrics = ["accuracy", "precision", "recall", "f1"]

    for group_name, gdf in groups.items():
        plt.figure(figsize=(8, 6))

        # Choose x labels based on type of experiment
        if group_name == "Learning Rate":
            x = gdf["lr"].astype(str)
        elif group_name == "Batch Size":
            x = gdf["batch_size"].astype(str)
        elif group_name == "Filters":
            x = gdf["filters"]
        elif group_name == "Layers":
            x = gdf["layers"].astype(str)

        # Plot each metric as a bar group
        bar_width = 0.2
        positions = range(len(x))

        for i, metric in enumerate(metrics):
            plt.bar(
                [p + i * bar_width for p in positions],
                gdf[metric],
                width=bar_width,
                label=metric.capitalize(),
            )

        # Ticks and labels
        plt.xticks([p + 1.5 * bar_width for p in positions], x, rotation=30)
        plt.ylim(0, 1.0)
        plt.title(f"{group_name} Ablation Results")
        plt.xlabel(group_name)
        plt.ylabel("Score")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save each chart
        save_path = os.path.join(out_dir, f"{group_name.lower().replace(' ', '_')}_ablation.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_ablation_results("cnn_results-20250911T213809Z-1-001\cnn_results\\ablation_summary.csv")
