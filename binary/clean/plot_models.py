import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ============================================================
# PLOT STYLE
# ============================================================

FONT_SIZE = 20
AXIS_LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 20

LINE_WIDTH = 3.4
FINE_TUNING_LINE_WIDTH = 1.5
AXES_LINE_WIDTH = 1.8
GRID_LINE_WIDTH = 1.2

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": AXIS_LABEL_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.linewidth": AXES_LINE_WIDTH,
    "lines.linewidth": LINE_WIDTH,
})

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "MobileNetV2": "mobilenet",
    "DenseNet121": "densenet",
    "ResNet50": "resnet",
    "VGG16": "vgg",
}

OUTPUT_DIR = os.path.join(BASE_DIR, "graficos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD HEAD + FINE-TUNING LOGS
# ============================================================

def load_combined_log(model_suffix):

    head_csv = os.path.join(
        BASE_DIR,
        f"training_log_head_{model_suffix}_clean.csv"
    )

    finetune_csv = os.path.join(
        BASE_DIR,
        f"training_log_finetune_{model_suffix}_clean.csv"
    )

    if not os.path.exists(head_csv):
        raise FileNotFoundError(head_csv)

    if not os.path.exists(finetune_csv):
        raise FileNotFoundError(finetune_csv)

    df_head = pd.read_csv(head_csv)
    df_ft = pd.read_csv(finetune_csv)

    df = pd.concat(
        [df_head, df_ft],
        ignore_index=True
    )

    # Continuous epochs: 1, 2, 3, ...
    df["epoch_plot"] = range(1, len(df) + 1)

    n_head_epochs = len(df_head)

    return df, n_head_epochs


# ============================================================
# LOAD ALL MODELS
# ============================================================

histories = {}
fine_tuning_epochs = {}

for model_name, model_suffix in MODELS.items():

    try:
        df, n_head = load_combined_log(model_suffix)

        histories[model_name] = df
        fine_tuning_epochs[model_name] = n_head + 0.5

    except FileNotFoundError as e:
        print(f"[WARN] {model_name}: file not found: {e}")


# ============================================================
# FIND COMMON FINE-TUNING TRANSITION
# ============================================================

if fine_tuning_epochs:

    transitions = list(fine_tuning_epochs.values())

    if len(set(transitions)) == 1:
        transition_epoch = transitions[0]
    else:
        transition_epoch = None

else:
    transition_epoch = None


# ============================================================
# FUNCTION TO CREATE COMPARISON PLOTS
# ============================================================

def plot_models(metric_options, ylabel, filename, ylim=None):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for model_name, df in histories.items():

        # Find the correct column name
        metric = None

        for option in metric_options:
            if option in df.columns:
                metric = option
                break

        if metric is None:
            print(
                f"[WARN] {model_name}: "
                f"none of {metric_options} found"
            )
            continue

        ax.plot(
            df["epoch_plot"],
            df[metric],
            linewidth=LINE_WIDTH,
            label=model_name
        )

    # Fine-tuning transition
    if transition_epoch is not None:

        ax.axvline(
            x=transition_epoch,
            linestyle="--",
            linewidth=FINE_TUNING_LINE_WIDTH,
            color="black",
            label="Start of fine-tuning"
        )

    # Axis labels
    ax.set_xlabel(
        "Number of Epochs",
        fontsize=AXIS_LABEL_SIZE
    )

    ax.set_ylabel(
        ylabel,
        fontsize=AXIS_LABEL_SIZE
    )

    # Axis limits
    ax.set_xlim(
        0,
        18
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    # X axis
    ax.xaxis.set_major_locator(
        MaxNLocator(
            integer=True
        )
    )

    # Y axis
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    # Tick appearance
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE,
        width=AXES_LINE_WIDTH,
        length=7
    )

    # Grid
    ax.grid(
        True,
        alpha=0.3,
        linewidth=GRID_LINE_WIDTH
    )

    # Legend
    ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=True,
        loc="best"
    )

    # Axis borders
    for spine in ax.spines.values():
        spine.set_linewidth(
            AXES_LINE_WIDTH
        )

    fig.tight_layout()

    # PDF
    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{filename}.pdf"
        ),
        bbox_inches="tight"
    )

    # PNG
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
# VALIDATION / MONITORING ACCURACY
# ============================================================

plot_models(
    metric_options=[
        "val_accuracy",
        "val_binary_accuracy"
    ],
    ylabel="Validation Accuracy",
    filename="binary_validation_accuracy_clean",
    ylim=(0.1, 1.05)
)


# ============================================================
# VALIDATION / MONITORING LOSS
# ============================================================

plot_models(
    metric_options=[
        "val_loss"
    ],
    ylabel="Validation Loss",
    filename="binary_validation_loss_clean"
)

print("All comparison plots saved.")