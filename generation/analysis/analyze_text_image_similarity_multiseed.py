import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Project imports
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "encoder")
)

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


# ============================================================
# Config
# ============================================================

MODEL_NAME = (
    "lightgcn_plus_gated_openai_clip_image"
)

DATASET_NAME = "amazon7425"

SEEDS = [2023, 2024, 2025]

CHECKPOINT_DIR = (
    ROOT /
    "encoder/checkpoint" /
    MODEL_NAME
)

ANALYSIS_DIR = (
    ROOT /
    "generation/analysis"
)

OUTPUT_DIR = (
    ANALYSIS_DIR /
    "text_image_similarity_multiseed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GATES_PATH = (
    ANALYSIS_DIR /
    "gated_openai_clip_image_amazon7425_"
    "multiseed_gates_by_seed.npy"
)


# ============================================================
# Helper
# ============================================================

def get_state_dict(checkpoint):

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        return checkpoint["state_dict"]

    return checkpoint


# ============================================================
# Load dataset / model
# ============================================================

print("=" * 70)
print("Text-Image Similarity Analysis")
print("OpenAI Text + CLIP Image - Three Seeds")
print("=" * 70)

data_handler = build_data_handler()
data_handler.load_data()

model = build_model(
    data_handler
)

model = model.to(
    configs["device"]
)


# ============================================================
# Extract projected cosine similarity per seed
# ============================================================

similarities = []

for seed in SEEDS:

    ckpt_path = (
        CHECKPOINT_DIR /
        f"{MODEL_NAME}-{DATASET_NAME}-{seed}.pth"
    )

    print(
        f"\nLoading seed {seed}:"
    )
    print(ckpt_path)

    checkpoint = torch.load(
        ckpt_path,
        map_location=configs["device"]
    )

    model.load_state_dict(
        get_state_dict(checkpoint)
    )

    model.eval()

    with torch.no_grad():

        text_proj = model.text_projection(
            model.item_text_embeds
        )

        image_proj = model.image_projection(
            model.item_image_embeds
        )

        sim = F.cosine_similarity(
            text_proj,
            image_proj,
            dim=1
        )

    sim = (
        sim
        .detach()
        .cpu()
        .numpy()
    )

    assert len(sim) == 7425
    assert np.isfinite(sim).all()

    similarities.append(sim)

    print(
        f"Similarity | "
        f"mean={sim.mean():.6f} | "
        f"std={sim.std():.6f} | "
        f"min={sim.min():.6f} | "
        f"max={sim.max():.6f}"
    )


similarities = np.stack(
    similarities,
    axis=0
)

similarity_mean = (
    similarities.mean(axis=0)
)

similarity_std = (
    similarities.std(
        axis=0,
        ddof=1
    )
)

print(
    "\nSimilarity matrix:",
    similarities.shape
)


# ============================================================
# Load gate weights
# ============================================================

gates = np.load(
    GATES_PATH
)

assert gates.shape == (
    3,
    7425
)

image_weights = (
    1.0 -
    gates
)

image_weight_mean = (
    image_weights.mean(axis=0)
)


# ============================================================
# Per-seed correlation
# ============================================================

print("\n" + "=" * 70)
print("PER-SEED CORRELATION")
print("=" * 70)

correlation_rows = []

for i, seed in enumerate(SEEDS):

    x = similarities[i]
    y = image_weights[i]

    pearson_r, pearson_p = pearsonr(
        x,
        y
    )

    spearman_r, spearman_p = spearmanr(
        x,
        y
    )

    print(
        f"{seed} | "
        f"Pearson={pearson_r:.6f} "
        f"(p={pearson_p:.3e}) | "
        f"Spearman={spearman_r:.6f} "
        f"(p={spearman_p:.3e})"
    )

    correlation_rows.append({
        "seed": seed,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    })


# ============================================================
# Three-seed mean correlation
# ============================================================

pearson_r, pearson_p = pearsonr(
    similarity_mean,
    image_weight_mean
)

spearman_r, spearman_p = spearmanr(
    similarity_mean,
    image_weight_mean
)

print("\n" + "=" * 70)
print("THREE-SEED MEAN CORRELATION")
print("=" * 70)

print(
    f"Pearson  r={pearson_r:.6f}, "
    f"p={pearson_p:.3e}"
)

print(
    f"Spearman rho={spearman_r:.6f}, "
    f"p={spearman_p:.3e}"
)


# ============================================================
# Quartile analysis
# ============================================================

df = pd.DataFrame({
    "item_id": np.arange(7425),

    "image_weight_mean":
        image_weight_mean,

    "similarity_mean":
        similarity_mean,

    "similarity_std":
        similarity_std,

    "gate_mean":
        1.0 - image_weight_mean,
})


df["image_weight_quartile"] = pd.qcut(
    df["image_weight_mean"],
    q=4,
    labels=[
        "Q1_lowest",
        "Q2",
        "Q3",
        "Q4_highest",
    ]
)


quartile_stats = (
    df
    .groupby(
        "image_weight_quartile",
        observed=True
    )
    .agg(
        n_items=(
            "item_id",
            "count"
        ),
        mean_image_weight=(
            "image_weight_mean",
            "mean"
        ),
        mean_similarity=(
            "similarity_mean",
            "mean"
        ),
        median_similarity=(
            "similarity_mean",
            "median"
        ),
        std_similarity=(
            "similarity_mean",
            "std"
        ),
    )
    .reset_index()
)


print("\n" + "=" * 70)
print("IMAGE-WEIGHT QUARTILE ANALYSIS")
print("=" * 70)

print(
    quartile_stats.to_string(
        index=False
    )
)


# ============================================================
# Top / Bottom 50 comparison
# ============================================================

order = np.argsort(
    image_weight_mean
)

bottom50 = order[:50]
top50 = order[-50:]

print("\n" + "=" * 70)
print("TOP / BOTTOM 50 SIMILARITY")
print("=" * 70)

print(
    "Top50 high image weight "
    "mean similarity:",
    similarity_mean[top50].mean()
)

print(
    "Bottom50 low image weight "
    "mean similarity:",
    similarity_mean[bottom50].mean()
)


# ============================================================
# Save
# ============================================================

np.save(
    OUTPUT_DIR /
    "text_image_similarity_by_seed.npy",
    similarities
)

np.save(
    OUTPUT_DIR /
    "text_image_similarity_mean.npy",
    similarity_mean
)

df.to_csv(
    OUTPUT_DIR /
    "item_text_image_similarity.csv",
    index=False
)

quartile_stats.to_csv(
    OUTPUT_DIR /
    "quartile_similarity_statistics.csv",
    index=False
)

pd.DataFrame(
    correlation_rows
).to_csv(
    OUTPUT_DIR /
    "per_seed_correlations.csv",
    index=False
)

print("\nSaved to:")
print(OUTPUT_DIR)

print(
    "\nText-image similarity analysis completed."
)