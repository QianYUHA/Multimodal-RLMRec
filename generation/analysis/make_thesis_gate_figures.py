from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ANALYSIS_DIR = Path("generation/analysis")

OUTPUT_DIR = ANALYSIS_DIR / "thesis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GATE_MEAN_PATH = (
    ANALYSIS_DIR /
    "gated_openai_clip_image_amazon7425_multiseed_gate_mean.npy"
)

GENRE_PATH = (
    ANALYSIS_DIR /
    "genre_multiseed/"
    "genre_image_weight_statistics_min30.csv"
)

DIST_PATH = (
    ANALYSIS_DIR /
    "visual_distinctiveness/"
    "item_visual_distinctiveness.csv"
)


# ============================================================
# Plot settings
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def save_figure(fig, name):
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    print("Saved:", png_path)
    print("Saved:", pdf_path)

    plt.close(fig)


# ============================================================
# Figure 1
# Distribution of three-seed averaged image weights
# ============================================================

gate_mean = np.load(
    GATE_MEAN_PATH
).reshape(-1)

image_weight = 1.0 - gate_mean

mean_weight = image_weight.mean()


fig, ax = plt.subplots(
    figsize=(6.4, 4.2)
)

ax.hist(
    image_weight,
    bins=35,
    alpha=0.85
)

ax.axvline(
    mean_weight,
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {mean_weight:.3f}"
)

ax.set_xlabel(
    "Three-seed averaged image weight"
)

ax.set_ylabel(
    "Number of items"
)

ax.set_title(
    "Distribution of Learned Image Weights"
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    alpha=0.2
)

fig.tight_layout()

save_figure(
    fig,
    "fig_gate_weight_distribution"
)


# ============================================================
# Figure 2
# Mean image weight by genre
# ============================================================

genre_df = pd.read_csv(
    GENRE_PATH
)

# Sort low -> high so high values appear at top
genre_df = genre_df.sort_values(
    "mean_image_weight",
    ascending=True
).copy()

# Approximate 95% CI of genre mean
genre_df["sem"] = (
    genre_df["std_image_weight"] /
    np.sqrt(genre_df["n_items"])
)

genre_df["ci95"] = (
    1.96 *
    genre_df["sem"]
)


fig, ax = plt.subplots(
    figsize=(7.2, 5.5)
)

ax.barh(
    genre_df["genre"],
    genre_df["mean_image_weight"],
    xerr=genre_df["ci95"],
    capsize=2
)

ax.axvline(
    mean_weight,
    linestyle="--",
    linewidth=1.2,
    label=f"Overall mean = {mean_weight:.3f}"
)

ax.set_xlabel(
    "Mean image weight"
)

ax.set_ylabel(
    "Genre"
)

ax.set_title(
    "Learned Image Weight by Genre"
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="x",
    alpha=0.2
)

# Add number of items after each bar
for i, row in genre_df.reset_index(
    drop=True
).iterrows():

    ax.text(
        row["mean_image_weight"] + 0.0005,
        i,
        f"n={int(row['n_items'])}",
        va="center",
        fontsize=8
    )


fig.tight_layout()

save_figure(
    fig,
    "fig_image_weight_by_genre"
)


# ============================================================
# Figure 3
# Visual distinctiveness by image-weight quartile
# ============================================================

dist_df = pd.read_csv(
    DIST_PATH
)

# Recreate quartile ordering explicitly
dist_df["quartile"] = pd.qcut(
    dist_df["image_weight_mean"],
    q=4,
    labels=[
        "Q1\nLowest",
        "Q2",
        "Q3",
        "Q4\nHighest",
    ]
)

quartile_df = (
    dist_df
    .groupby(
        "quartile",
        observed=True
    )
    .agg(
        n=(
            "item_id",
            "count"
        ),
        mean_distinctiveness=(
            "visual_distinctiveness",
            "mean"
        ),
        std_distinctiveness=(
            "visual_distinctiveness",
            "std"
        ),
    )
    .reset_index()
)

quartile_df["sem"] = (
    quartile_df["std_distinctiveness"] /
    np.sqrt(quartile_df["n"])
)

quartile_df["ci95"] = (
    1.96 *
    quartile_df["sem"]
)


fig, ax = plt.subplots(
    figsize=(6.2, 4.3)
)

bars = ax.bar(
    quartile_df["quartile"].astype(str),
    quartile_df["mean_distinctiveness"],
    yerr=quartile_df["ci95"],
    capsize=3
)

ax.set_xlabel(
    "Image-weight quartile"
)

ax.set_ylabel(
    "Mean visual distinctiveness"
)

ax.set_title(
    "Visual Distinctiveness Across Image-Weight Quartiles"
)

ax.grid(
    axis="y",
    alpha=0.2
)


# Add numeric values above bars
for bar, value in zip(
    bars,
    quartile_df["mean_distinctiveness"]
):

    ax.text(
        bar.get_x() +
        bar.get_width() / 2,

        value + 0.002,

        f"{value:.3f}",

        ha="center",
        va="bottom",
        fontsize=9
    )


fig.tight_layout()

save_figure(
    fig,
    "fig_distinctiveness_by_image_weight_quartile"
)


print("\nAll quantitative thesis figures completed.")
print("Output directory:", OUTPUT_DIR)