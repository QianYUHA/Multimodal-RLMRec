import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Config
# ============================================================

IMAGE_EMBED_PATH = Path(
    "data/amazon7425/item_image_clip.pkl"
)

GATE_MEAN_PATH = Path(
    "generation/analysis/"
    "gated_openai_clip_image_amazon7425_multiseed_gate_mean.npy"
)

OUTPUT_DIR = Path(
    "generation/analysis/visual_distinctiveness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Number of nearest visual neighbours
K = 10

# Process cosine similarity in chunks to avoid
# constructing the full 7425 x 7425 matrix at once.
CHUNK_SIZE = 512


# ============================================================
# 1. Load CLIP image embeddings
# ============================================================

print("=" * 70)
print("Visual Distinctiveness Analysis")
print("=" * 70)

print("\n[1] Loading CLIP image embeddings...")

with open(
    IMAGE_EMBED_PATH,
    "rb"
) as f:
    image_embeds = pickle.load(f)

image_embeds = np.asarray(
    image_embeds,
    dtype=np.float32
)

print(
    "Image embedding shape:",
    image_embeds.shape
)

assert image_embeds.shape == (
    7425,
    512
), (
    f"Unexpected image embedding shape: "
    f"{image_embeds.shape}"
)

assert np.isfinite(
    image_embeds
).all(), (
    "Image embeddings contain NaN or Inf."
)


# ============================================================
# 2. Check and normalize embeddings
# ============================================================

print("\n[2] Checking embedding norms...")

norms = np.linalg.norm(
    image_embeds,
    axis=1
)

zero_vectors = np.sum(
    norms == 0
)

print(
    "Zero vectors:",
    zero_vectors
)

assert zero_vectors == 0, (
    "Zero image embeddings detected."
)

normalized_embeds = (
    image_embeds /
    norms[:, None]
).astype(
    np.float32
)

print(
    "PASS: embeddings normalized."
)


# ============================================================
# 3. Find K nearest visual neighbours
# ============================================================

print(
    f"\n[3] Finding {K} nearest visual neighbours "
    "using chunked NumPy cosine similarity..."
)

N = normalized_embeds.shape[0]

neighbor_indices = np.empty(
    (N, K),
    dtype=np.int32
)

neighbor_similarities = np.empty(
    (N, K),
    dtype=np.float32
)


for start in range(
    0,
    N,
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        N
    )

    print(
        f"Processing items "
        f"{start:4d}-{end - 1:4d} "
        f"/ {N - 1}"
    )

    # normalized dot product = cosine similarity
    similarities = (
        normalized_embeds[start:end]
        @ normalized_embeds.T
    )

    # --------------------------------------------------------
    # Remove self-similarity
    # --------------------------------------------------------

    local_rows = np.arange(
        end - start
    )

    global_cols = np.arange(
        start,
        end
    )

    similarities[
        local_rows,
        global_cols
    ] = -np.inf

    # --------------------------------------------------------
    # Find K highest similarities
    # --------------------------------------------------------

    topk_idx = np.argpartition(
        -similarities,
        kth=K - 1,
        axis=1
    )[:, :K]

    topk_sim = np.take_along_axis(
        similarities,
        topk_idx,
        axis=1
    )

    # Sort selected K neighbours:
    # highest cosine similarity first
    order = np.argsort(
        -topk_sim,
        axis=1
    )

    topk_idx = np.take_along_axis(
        topk_idx,
        order,
        axis=1
    )

    topk_sim = np.take_along_axis(
        topk_sim,
        order,
        axis=1
    )

    neighbor_indices[
        start:end
    ] = topk_idx

    neighbor_similarities[
        start:end
    ] = topk_sim


print(
    "\nNearest-neighbour index shape:",
    neighbor_indices.shape
)

print(
    "Nearest-neighbour similarity shape:",
    neighbor_similarities.shape
)

assert neighbor_indices.shape == (
    7425,
    K
)

assert neighbor_similarities.shape == (
    7425,
    K
)

assert np.isfinite(
    neighbor_similarities
).all()


# ============================================================
# 4. Compute visual distinctiveness
# ============================================================

print(
    "\n[4] Computing visual distinctiveness..."
)

# Average similarity to the K closest visual neighbours
mean_neighbor_similarity = (
    neighbor_similarities.mean(
        axis=1
    )
)

# Higher value = visually more distinctive
distinctiveness = (
    1.0 -
    mean_neighbor_similarity
)


print(
    "\nDistinctiveness statistics"
)

print("-" * 50)

print(
    "N      :",
    len(distinctiveness)
)

print(
    "Mean   :",
    distinctiveness.mean()
)

print(
    "Std    :",
    distinctiveness.std()
)

print(
    "Min    :",
    distinctiveness.min()
)

print(
    "25%    :",
    np.percentile(
        distinctiveness,
        25
    )
)

print(
    "Median :",
    np.median(
        distinctiveness
    )
)

print(
    "75%    :",
    np.percentile(
        distinctiveness,
        75
    )
)

print(
    "Max    :",
    distinctiveness.max()
)


# ============================================================
# 5. Load three-seed mean gate
# ============================================================

print(
    "\n[5] Loading three-seed mean gate..."
)

gate_mean = np.load(
    GATE_MEAN_PATH
).reshape(-1)

assert len(
    gate_mean
) == 7425

assert np.isfinite(
    gate_mean
).all()

image_weight_mean = (
    1.0 -
    gate_mean
)


print(
    "Mean image weight:",
    image_weight_mean.mean()
)


# ============================================================
# 6. Correlation:
#    image weight vs visual distinctiveness
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "IMAGE WEIGHT vs VISUAL DISTINCTIVENESS"
)

print(
    "=" * 70
)


pearson_r, pearson_p = pearsonr(
    distinctiveness,
    image_weight_mean
)

spearman_r, spearman_p = spearmanr(
    distinctiveness,
    image_weight_mean
)


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
# 7. Build item-level dataframe
# ============================================================

df = pd.DataFrame({
    "item_id":
        np.arange(7425),

    "image_weight_mean":
        image_weight_mean,

    "mean_neighbor_similarity":
        mean_neighbor_similarity,

    "visual_distinctiveness":
        distinctiveness,
})


# ============================================================
# 8. Image-weight quartile analysis
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

        mean_neighbor_similarity=(
            "mean_neighbor_similarity",
            "mean"
        ),

        mean_distinctiveness=(
            "visual_distinctiveness",
            "mean"
        ),

        median_distinctiveness=(
            "visual_distinctiveness",
            "median"
        ),

        std_distinctiveness=(
            "visual_distinctiveness",
            "std"
        ),
    )
    .reset_index()
)


print(
    "\n" + "=" * 70
)

print(
    "IMAGE-WEIGHT QUARTILE ANALYSIS"
)

print(
    "=" * 70
)

print(
    quartile_stats.to_string(
        index=False
    )
)


# ============================================================
# 9. Top / Bottom 50 comparison
# ============================================================

order = np.argsort(
    image_weight_mean
)

bottom50 = order[:50]

top50 = order[-50:][::-1]


top50_mean_distinctiveness = (
    distinctiveness[
        top50
    ].mean()
)

bottom50_mean_distinctiveness = (
    distinctiveness[
        bottom50
    ].mean()
)


print(
    "\n" + "=" * 70
)

print(
    "TOP / BOTTOM 50 DISTINCTIVENESS"
)

print(
    "=" * 70
)

print(
    "Top50 high image weight "
    "mean distinctiveness:",
    top50_mean_distinctiveness
)

print(
    "Bottom50 low image weight "
    "mean distinctiveness:",
    bottom50_mean_distinctiveness
)


print(
    "\nTop50 high image weight "
    "mean neighbour similarity:",
    mean_neighbor_similarity[
        top50
    ].mean()
)

print(
    "Bottom50 low image weight "
    "mean neighbour similarity:",
    mean_neighbor_similarity[
        bottom50
    ].mean()
)


# ============================================================
# 10. Save item-level results
# ============================================================

item_output_path = (
    OUTPUT_DIR /
    "item_visual_distinctiveness.csv"
)

df.to_csv(
    item_output_path,
    index=False
)


# ============================================================
# 11. Save quartile statistics
# ============================================================

quartile_output_path = (
    OUTPUT_DIR /
    "quartile_distinctiveness_statistics.csv"
)

quartile_stats.to_csv(
    quartile_output_path,
    index=False
)


# ============================================================
# 12. Save nearest-neighbour arrays
# ============================================================

indices_output_path = (
    OUTPUT_DIR /
    "nearest_neighbor_indices.npy"
)

similarities_output_path = (
    OUTPUT_DIR /
    "nearest_neighbor_similarities.npy"
)

np.save(
    indices_output_path,
    neighbor_indices
)

np.save(
    similarities_output_path,
    neighbor_similarities
)


# ============================================================
# 13. Save top / bottom 50 tables
# ============================================================

top50_df = df.iloc[
    top50
].copy()

top50_df.insert(
    0,
    "rank",
    np.arange(
        1,
        len(top50_df) + 1
    )
)

bottom50_df = df.iloc[
    bottom50
].copy()

bottom50_df.insert(
    0,
    "rank",
    np.arange(
        1,
        len(bottom50_df) + 1
    )
)


top50_df.to_csv(
    OUTPUT_DIR /
    "top50_high_image_weight_distinctiveness.csv",
    index=False
)

bottom50_df.to_csv(
    OUTPUT_DIR /
    "bottom50_low_image_weight_distinctiveness.csv",
    index=False
)


# ============================================================
# 14. Summary
# ============================================================

print(
    "\nSaved outputs:"
)

print(
    item_output_path
)

print(
    quartile_output_path
)

print(
    indices_output_path
)

print(
    similarities_output_path
)

print(
    OUTPUT_DIR /
    "top50_high_image_weight_distinctiveness.csv"
)

print(
    OUTPUT_DIR /
    "bottom50_low_image_weight_distinctiveness.csv"
)


print(
    "\n" + "=" * 70
)

print(
    "Visual distinctiveness analysis completed."
)

print(
    "=" * 70
)