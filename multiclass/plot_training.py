import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 24
})

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_KEYS = [
    "mobilevit_xxs",
    "tf_efficientnet_lite1",
    "mobilenetv2_100",
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
]

MODEL_NAMES = {
    "mobilevit_xxs": "MobileViT-XXS",
    "tf_efficientnet_lite1": "EfficientNet-Lite1",
    "mobilenetv2_100": "MobileNetV2",
    "mobilenetv3_small_100": "MobileNetV3-Small",
    "mobilenetv3_large_100": "MobileNetV3-Large",
}

TRAINING_FILES = {
    "mobilevit_xxs": "training_history_mobilevit_xxs.csv",
    "tf_efficientnet_lite1": "training_history_efficientnetlite1.csv",
    "mobilenetv2_100": "training_history_mobilenetv2_100.csv",
    "mobilenetv3_small_100": "training_history_mobilenetv3_small_100.csv",
    "mobilenetv3_large_100": "training_history_mobilenetv3_large_100.csv",
}

OUTPUT_DIR = "graficos"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD TRAINING HISTORIES
# ============================================================

histories = {}

for model_key in MODEL_KEYS:

    training_csv = TRAINING_FILES[model_key]

    if not os.path.exists(training_csv):
        print("Training file not found:", training_csv)
        continue

    histories[model_key] = pd.read_csv(training_csv)

# ============================================================
# FUNCTION TO CREATE COMPARISON PLOTS
# ============================================================

def plot_models(metric, ylabel, filename, ylim=None):

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    for model_key in MODEL_KEYS:

        if model_key not in histories:
            continue

        df = histories[model_key]

        ax.plot(
            df["epoch"],
            df[metric],
            linewidth=2.5,
            label=MODEL_NAMES[model_key]
        )

    # Start of fine-tuning after epoch 10
    ax.axvline(
        x=10.5,
        linestyle="--",
        linewidth=1.3,
        color="black",
        label="Start of fine-tuning"
    )

    # Axis labels
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(ylabel)

    # X-axis
    ax.set_xlim(0, 62)
    ax.set_xticks(
        range(0, 61, 10)
    )

    # Y-axis
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    # Grid
    ax.grid(
        True,
        alpha=0.3
    )

    # Legend
    ax.legend()

    fig.tight_layout()

    # Save PDF
    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{filename}.pdf"
        ),
        bbox_inches="tight"
    )

    # Save PNG
    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{filename}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================
# 1. VALIDATION ACCURACY
# ============================================================

plot_models(
    metric="val_acc",
    ylabel="Validation Accuracy",
    filename="comparison_validation_accuracy",
    ylim=(0.3, 1.0)
)

# ============================================================
# 2. VALIDATION LOSS
# ============================================================

plot_models(
    metric="val_loss",
    ylabel="Validation Loss",
    filename="comparison_validation_loss",
    ylim=(0, 2.5)
)

# ============================================================
# 3. TRAINING ACCURACY
# ============================================================

plot_models(
    metric="train_acc",
    ylabel="Training Accuracy",
    filename="comparison_training_accuracy",
    ylim=(0.2, 1.05)
)

# ============================================================
# 4. TRAINING LOSS
# ============================================================

plot_models(
    metric="train_loss",
    ylabel="Training Loss",
    filename="comparison_training_loss",
    ylim=(-0.05, 3.5)
)

# ============================================================
# FINISHED
# ============================================================

print("All comparison plots saved.")