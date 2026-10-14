import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Paths
# ============================================================

TEXT_PATH = Path(
    "data/amazon7425/item_text_clip.pkl"
)

IMAGE_PATH = Path(
    "data/amazon7425/item_image_clip.pkl"
)

GATE_MEAN_PATH = Path(
    "generation/analysis/"
    "gated_openai_clip_image_amazon7425_multiseed_gate_mean.npy"
)

OUTPUT_DIR = Path(
    "generation/analysis/raw_clip_similarity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Load embeddings
# ============================================================

print("=" * 70)
print("Raw CLIP Text-Image Similarity Analysis")
print("=" * 70)

print("\n[1] Loading CLIP text embeddings...")

with open(TEXT_PATH, "rb") as f:
    text_embeds = pickle.load(f)

text_embeds = np.asarray(
    text_embeds,
    dtype=np.float32
)

print(
    "Text shape:",
    text_embeds.shape
)


print("\n[2] Loading CLIP image embeddings...")

with open(IMAGE_PATH, "rb") as f:
    image_embeds = pickle.load(f)

image_embeds = np.asarray(
    image_embeds,
    dtype=np.float32
)

print(
    "Image shape:",
    image_embeds.shape
)


assert text_embeds.shape == (7425, 512)
assert image_embeds.shape == (7425, 512)

assert np.isfinite(text_embeds).all()
assert np.isfinite(image_embeds).all()


# ============================================================
# 2. Check zero vectors
# ============================================================

text_norms = np.linalg.norm(
    text_embeds,
    axis=1
)

image_norms = np.linalg.norm(
    image_embeds,
    axis=1
)


print("\n[3] Embedding checks")

print(
    "Zero text vectors :",
    np.sum(text_norms == 0)
)

print(
    "Zero image vectors:",
    np.sum(image_norms == 0)
)


assert np.all(text_norms > 0)
assert np.all(image_norms > 0)


# ============================================================
# 3. Normalize
# ============================================================

text_normed = (
    text_embeds /
    text_norms[:, None]
)

image_normed = (
    image_embeds /
    image_norms[:, None]
)


# ============================================================
# 4. Row-wise CLIP cosine similarity
# ============================================================

clip_similarity = np.sum(
    text_normed *
    image_normed,
    axis=1
)


print("\n[4] Raw CLIP text-image similarity")

print("-" * 50)

print(
    "N      :",
    len(clip_similarity)
)

print(
    "Mean   :",
    clip_similarity.mean()
)

print(
    "Std    :",
    clip_similarity.std()
)

print(
    "Min    :",
    clip_similarity.min()
)

print(
    "25%    :",
    np.percentile(
        clip_similarity,
        25
    )
)

print(
    "Median :",
    np.median(
        clip_similarity
    )
)

print(
    "75%    :",
    np.percentile(
        clip_similarity,
        75
    )
)

print(
    "Max    :",
    clip_similarity.max()
)


# ============================================================
# 5. Load mean image weight
# ============================================================

gate_mean = np.load(
    GATE_MEAN_PATH
).reshape(-1)

assert len(gate_mean) == 7425

image_weight_mean = (
    1.0 -
    gate_mean
)


# ============================================================
# 6. Correlation
# ============================================================

pearson_r, pearson_p = pearsonr(
    clip_similarity,
    image_weight_mean
)

spearman_r, spearman_p = spearmanr(
    clip_similarity,
    image_weight_mean
)


print("\n" + "=" * 70)
print("IMAGE WEIGHT vs RAW CLIP TEXT-IMAGE SIMILARITY")
print("=" * 70)

print(
    f"Pearson r="
    f"{pearson_r:.6f}, "
    f"p={pearson_p:.3e}"
)

print(
    f"Spearman rho="
    f"{spearman_r:.6f}, "
    f"p={spearman_p:.3e}"
)


# ============================================================
# 7. Build dataframe
# ============================================================

df = pd.DataFrame({
    "item_id":
        np.arange(7425),

    "image_weight_mean":
        image_weight_mean,

    "clip_text_image_similarity":
        clip_similarity,
})


# ============================================================
# 8. Quartile analysis
# ============================================================

df[
    "image_weight_quartile"
] = pd.qcut(
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

        mean_clip_similarity=(
            "clip_text_image_similarity",
            "mean"
        ),

        median_clip_similarity=(
            "clip_text_image_similarity",
            "median"
        ),

        std_clip_similarity=(
            "clip_text_image_similarity",
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
# 9. Top / Bottom 50
# ============================================================

order = np.argsort(
    image_weight_mean
)

bottom50 = order[:50]

top50 = order[-50:][::-1]


print("\n" + "=" * 70)
print("TOP / BOTTOM 50 RAW CLIP SIMILARITY")
print("=" * 70)

print(
    "Top50 high image weight:",
    clip_similarity[
        top50
    ].mean()
)

print(
    "Bottom50 low image weight:",
    clip_similarity[
        bottom50
    ].mean()
)


# ============================================================
# 10. Save
# ============================================================

df.to_csv(
    OUTPUT_DIR /
    "item_raw_clip_similarity.csv",
    index=False
)

quartile_stats.to_csv(
    OUTPUT_DIR /
    "quartile_raw_clip_similarity.csv",
    index=False
)

np.save(
    OUTPUT_DIR /
    "raw_clip_text_image_similarity.npy",
    clip_similarity
)


print("\nSaved outputs:")

for p in sorted(
    OUTPUT_DIR.iterdir()
):
    print(p)


print("\n" + "=" * 70)
print("Raw CLIP similarity analysis completed.")
print("=" * 70)