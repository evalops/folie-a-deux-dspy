
import csv
import sys
import matplotlib.pyplot as plt


def main(csv_path, out_path="results/figures/tradeoff.png"):
    alphas, truth, agreement = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alphas.append(float(row["alpha"]))
            truth.append(float(row["truth"]))
            agreement.append(float(row["agreement"]))

    plt.figure()
    plt.plot(alphas, truth, marker="o", label="Truth accuracy")
    plt.plot(alphas, agreement, marker="s", label="Agreement rate")
    plt.xlabel("alpha (weight on Truth)")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Truth vs Agreement across alpha")
    plt.tight_layout()

    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_tradeoff.py results_alpha.csv [out_path]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_path = (
        sys.argv[2] if len(sys.argv) > 2 else "results/figures/tradeoff.png"
    )
    main(csv_path, out_path)
