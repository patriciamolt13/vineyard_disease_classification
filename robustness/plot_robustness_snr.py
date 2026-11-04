import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 15
})

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "robustness_noise_tese"

MODEL_KEYS = [
    "efficientnetlite1",
    "mobilenetv2",
    "mobilenetv3_small",
    "mobilenetv3_large",
]

MODEL_NAMES = {
    "efficientnetlite1": "EfficientNet-Lite1",
    "mobilenetv2": "MobileNetV2",
    "mobilenetv3_small": "MobileNetV3-Small",
    "mobilenetv3_large": "MobileNetV3-Large",
}

MODES = [
    "early",
    "mid",
    "late",
    "classifier",
    "depthwise",
    "full",
]

MODEL_STYLES = {
    "efficientnetlite1": {"marker": "o", "linestyle": "-"},
    "mobilenetv2": {"marker": "s", "linestyle": "-"},
    "mobilenetv3_small": {"marker": "^", "linestyle": "-"},
    "mobilenetv3_large": {"marker": "D", "linestyle": "-"},
}

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "comparison_plots_tese"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD ALL RESULTS
# ============================================================

all_results = []

for model_key in MODEL_KEYS:

    print(f"Loading model: {model_key}")

    summary_csv = os.path.join(
        BASE_DIR,
        model_key,
        f"{model_key}_uniform_noise_summary.csv"
    )

    if not os.path.exists(summary_csv):
        print("File not found:", summary_csv)
        continue

    df = pd.read_csv(summary_csv)

    required_columns = [
        "mode",
        "epsilon",
        "snr_db_mean",
        "accuracy_mean",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {summary_csv}: {missing_columns}"
        )

    df["model_key"] = model_key

    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    df["snr_db_mean"] = pd.to_numeric(df["snr_db_mean"], errors="coerce")
    df["accuracy_mean"] = pd.to_numeric(df["accuracy_mean"], errors="coerce")

    # Remove epsilon = 0, no finite SNR value is reported
    df = df[df["epsilon"] > 0].copy()

    # Keep only finite SNR values
    df = df[np.isfinite(df["snr_db_mean"])].copy()

    all_results.append(df)


if len(all_results) == 0:
    raise RuntimeError("No summary files were loaded.")

all_df = pd.concat(all_results, ignore_index=True)

# ============================================================
# CREATE ONE PLOT PER LAYER GROUP
# ============================================================

for mode in MODES:

    print(f"\nCreating plot for mode: {mode}")

    mode_df = all_df[all_df["mode"] == mode].copy()

    if mode_df.empty:
        print("No data for mode:", mode)
        continue

    fig, ax = plt.subplots(figsize=(6, 4))

    for model_key in MODEL_KEYS:

        subset = mode_df[mode_df["model_key"] == model_key].copy()

        if subset.empty:
            print("No data for:", model_key, mode)
            continue

        # Lower SNR = stronger perturbation
        # Higher SNR = weaker perturbation
        subset = subset.sort_values("snr_db_mean", ascending=True)

        style = MODEL_STYLES[model_key]

        ax.plot(
            subset["snr_db_mean"],
            subset["accuracy_mean"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            label=MODEL_NAMES[model_key],
        )

    ax.set_xlabel("Weight-level SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.19, 1)

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    file_base = f"{mode}_snr_db"

    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{file_base}.pdf"),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{file_base}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", os.path.join(OUTPUT_DIR, f"{file_base}.png"))


print("\nAll comparison plots saved.")