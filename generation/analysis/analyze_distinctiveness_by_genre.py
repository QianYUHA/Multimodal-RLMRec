from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Paths
# ============================================================

DIST_PATH = Path(
    "generation/analysis/"
    "visual_distinctiveness/"
    "item_visual_distinctiveness.csv"
)

GENRE_PATH = Path(
    "generation/analysis/"
    "genre_multiseed/"
    "item_genre_corrected_multiseed_gate.csv"
)

OUTPUT_DIR = Path(
    "generation/analysis/"
    "visual_distinctiveness/"
    "genre_control"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Load data
# ============================================================

print("=" * 70)
print("Within-Genre Visual Distinctiveness Analysis")
print("=" * 70)

dist_df = pd.read_csv(
    DIST_PATH
)

genre_df = pd.read_csv(
    GENRE_PATH
)


# Only need item_id + genre
genre_df = (
    genre_df[
        ["item_id", "genre"]
    ]
    .drop_duplicates()
)


merged = genre_df.merge(
    dist_df[
        [
            "item_id",
            "image_weight_mean",
            "visual_distinctiveness",
        ]
    ],
    on="item_id",
    how="inner"
)


print(
    "\nMerged rows:",
    len(merged)
)

print(
    "Unique items:",
    merged["item_id"].nunique()
)

print(
    "Genres:",
    merged["genre"].nunique()
)


# ============================================================
# 2. Calculate within-genre correlations
# ============================================================

MIN_ITEMS = 30

rows = []

for genre, group in merged.groupby(
    "genre"
):

    n = group["item_id"].nunique()

    if n < MIN_ITEMS:
        continue

    x = group[
        "visual_distinctiveness"
    ]

    y = group[
        "image_weight_mean"
    ]

    pearson_r, pearson_p = pearsonr(
        x,
        y
    )

    spearman_r, spearman_p = spearmanr(
        x,
        y
    )

    rows.append({
        "genre": genre,
        "n_items": n,

        "mean_image_weight":
            y.mean(),

        "mean_distinctiveness":
            x.mean(),

        "pearson_r":
            pearson_r,

        "pearson_p":
            pearson_p,

        "spearman_rho":
            spearman_r,

        "spearman_p":
            spearman_p,
    })


result = pd.DataFrame(
    rows
)

result = result.sort_values(
    "spearman_rho"
)


# ============================================================
# 3. Print
# ============================================================

print(
    "\nWITHIN-GENRE CORRELATIONS"
)

print("-" * 100)

print(
    result[
        [
            "genre",
            "n_items",
            "mean_image_weight",
            "mean_distinctiveness",
            "pearson_r",
            "spearman_rho",
            "spearman_p",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 4. Simple summary
# ============================================================

negative = (
    result["spearman_rho"] < 0
).sum()

positive = (
    result["spearman_rho"] > 0
).sum()

significant_negative = (
    (
        result["spearman_rho"] < 0
    )
    &
    (
        result["spearman_p"] < 0.05
    )
).sum()


print("\nSummary")
print("-" * 50)

print(
    "Genres with negative Spearman:",
    negative,
    "/",
    len(result)
)

print(
    "Genres with positive Spearman:",
    positive,
    "/",
    len(result)
)

print(
    "Genres with significant "
    "negative Spearman (p<0.05):",
    significant_negative,
)


# ============================================================
# 5. Save
# ============================================================

output_path = (
    OUTPUT_DIR /
    "within_genre_distinctiveness_correlations.csv"
)

result.to_csv(
    output_path,
    index=False
)

print(
    "\nSaved:",
    output_path
)

print(
    "\nWithin-genre analysis completed."
)