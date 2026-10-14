from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ANALYSIS_DIR = Path("generation/analysis")

GENRE_PATH = (
    ANALYSIS_DIR /
    "genre_gate_item_tags.csv"
)

GATE_MEAN_PATH = (
    ANALYSIS_DIR /
    "gated_openai_clip_image_amazon7425_multiseed_gate_mean.npy"
)

GATE_STD_PATH = (
    ANALYSIS_DIR /
    "gated_openai_clip_image_amazon7425_multiseed_gate_std.npy"
)

TOP50_PATH = (
    ANALYSIS_DIR /
    "gated_openai_clip_image_amazon7425_multiseed_top50_image_weight.csv"
)

BOTTOM50_PATH = (
    ANALYSIS_DIR /
    "gated_openai_clip_image_amazon7425_multiseed_bottom50_image_weight.csv"
)

OUTPUT_DIR = (
    ANALYSIS_DIR /
    "genre_multiseed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Load genre tags
# ============================================================

genre_df = pd.read_csv(GENRE_PATH)

# Only reuse genre labels.
genre_df = genre_df[
    ["item_id", "genre"]
].copy()

genre_df = genre_df.dropna(
    subset=["item_id", "genre"]
)

genre_df["item_id"] = (
    genre_df["item_id"]
    .astype(int)
)

genre_df["genre"] = (
    genre_df["genre"]
    .astype(str)
    .str.strip()
)

# Remove accidental duplicate item-genre pairs
genre_df = genre_df.drop_duplicates(
    subset=["item_id", "genre"]
)


print("=" * 70)
print("Genre Analysis - Corrected Multi-seed Gates")
print("=" * 70)

print("\nGenre rows:", len(genre_df))
print(
    "Unique tagged items:",
    genre_df["item_id"].nunique()
)
print(
    "Unique genres:",
    genre_df["genre"].nunique()
)

print(
    "Item coverage:",
    f"{genre_df['item_id'].nunique() / 7425 * 100:.2f}%"
)


# ============================================================
# 2. Load corrected three-seed gate values
# ============================================================

gate_mean = np.load(
    GATE_MEAN_PATH
).reshape(-1)

gate_std = np.load(
    GATE_STD_PATH
).reshape(-1)

assert len(gate_mean) == 7425
assert len(gate_std) == 7425


item_gate_df = pd.DataFrame({
    "item_id": np.arange(7425),
    "gate_mean": gate_mean,
    "gate_std": gate_std,
})

item_gate_df[
    "image_weight_mean"
] = (
    1.0 -
    item_gate_df["gate_mean"]
)

item_gate_df[
    "image_weight_std"
] = item_gate_df["gate_std"]


# ============================================================
# 3. Merge genre + corrected gate
# ============================================================

merged = genre_df.merge(
    item_gate_df,
    on="item_id",
    how="inner"
)

print(
    "\nMerged rows:",
    len(merged)
)

print(
    "Merged unique items:",
    merged["item_id"].nunique()
)


# ============================================================
# 4. Full-dataset genre statistics
# ============================================================

genre_stats = (
    merged
    .groupby("genre")
    .agg(
        n_items=("item_id", "nunique"),
        mean_image_weight=(
            "image_weight_mean",
            "mean"
        ),
        median_image_weight=(
            "image_weight_mean",
            "median"
        ),
        std_image_weight=(
            "image_weight_mean",
            "std"
        ),
        mean_gate_std=(
            "gate_std",
            "mean"
        ),
    )
    .reset_index()
)

# Sort by mean image weight
genre_stats = genre_stats.sort_values(
    "mean_image_weight",
    ascending=False
)

genre_stats.to_csv(
    OUTPUT_DIR /
    "genre_image_weight_statistics.csv",
    index=False
)


# ============================================================
# 5. More reliable table for sufficiently large genres
# ============================================================

# Avoid over-interpreting genres represented
# by only a handful of books.
MIN_ITEMS = 30

genre_stats_filtered = (
    genre_stats[
        genre_stats["n_items"] >= MIN_ITEMS
    ]
    .copy()
)

genre_stats_filtered.to_csv(
    OUTPUT_DIR /
    "genre_image_weight_statistics_min30.csv",
    index=False
)


print(
    "\nTOP GENRES BY MEAN IMAGE WEIGHT "
    f"(n >= {MIN_ITEMS})"
)
print("-" * 70)

print(
    genre_stats_filtered[
        [
            "genre",
            "n_items",
            "mean_image_weight",
            "median_image_weight",
            "std_image_weight",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


print(
    "\nLOWEST GENRES BY MEAN IMAGE WEIGHT "
    f"(n >= {MIN_ITEMS})"
)
print("-" * 70)

print(
    genre_stats_filtered[
        [
            "genre",
            "n_items",
            "mean_image_weight",
            "median_image_weight",
            "std_image_weight",
        ]
    ]
    .tail(15)
    .sort_values(
        "mean_image_weight"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 6. Top50 / Bottom50 genre composition
# ============================================================

top50 = pd.read_csv(
    TOP50_PATH
)

bottom50 = pd.read_csv(
    BOTTOM50_PATH
)

# The multiseed files created earlier use item_id.
# Handle either naming style in case files differ.
def get_item_id_column(df):

    if "item_id" in df.columns:
        return "item_id"

    if "amazon7425_id" in df.columns:
        return "amazon7425_id"

    raise ValueError(
        "Could not find item ID column."
    )


top_col = get_item_id_column(top50)
bottom_col = get_item_id_column(bottom50)

top_ids = set(
    top50[top_col]
    .astype(int)
)

bottom_ids = set(
    bottom50[bottom_col]
    .astype(int)
)


def extreme_genre_table(
    ids,
    group_name
):

    subset = genre_df[
        genre_df["item_id"].isin(ids)
    ]

    counts = (
        subset
        .groupby("genre")["item_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
        .reset_index(
            name="item_count"
        )
    )

    # Fraction of the 50 books carrying this tag.
    # Because genre is multi-label, percentages
    # do NOT need to sum to 100%.
    counts[
        "item_percentage"
    ] = (
        counts["item_count"] /
        len(ids) *
        100
    )

    counts[
        "group"
    ] = group_name

    return counts


top_genres = extreme_genre_table(
    top_ids,
    "Top 50 high image weight"
)

bottom_genres = extreme_genre_table(
    bottom_ids,
    "Bottom 50 low image weight"
)

top_genres.to_csv(
    OUTPUT_DIR /
    "top50_genre_composition.csv",
    index=False
)

bottom_genres.to_csv(
    OUTPUT_DIR /
    "bottom50_genre_composition.csv",
    index=False
)


print(
    "\nTOP 50 HIGH-IMAGE-WEIGHT GENRES"
)
print("-" * 70)

print(
    top_genres
    .head(15)
    .to_string(index=False)
)


print(
    "\nBOTTOM 50 LOW-IMAGE-WEIGHT GENRES"
)
print("-" * 70)

print(
    bottom_genres
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 7. Save corrected item-genre table
# ============================================================

merged.to_csv(
    OUTPUT_DIR /
    "item_genre_corrected_multiseed_gate.csv",
    index=False
)


print("\nSaved outputs:")
for p in sorted(
    OUTPUT_DIR.glob("*.csv")
):
    print(p)

print("\nGenre analysis completed.")