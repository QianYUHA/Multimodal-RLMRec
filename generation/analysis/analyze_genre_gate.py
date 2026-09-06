import os
import csv
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ITEM_PROFILE_PATH = "data/amazon7425/itm_prf.pkl"

CLIP_GATE_PATH = (
    "generation/analysis/"
    "gated_clip_amazon7425_gate.npy"
)

OPENAI_GATE_PATH = (
    "generation/analysis/"
    "gated_openai_clip_image_amazon7425_gate.npy"
)

OUTPUT_DIR = "generation/analysis"

ITEM_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "genre_gate_item_tags.csv"
)

SUMMARY_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "genre_gate_comparison.csv"
)

FIGURE_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "genre_gate_comparison.png"
)


# ============================================================
# Genre dictionary
#
# Genre labels are inferred from profile/reasoning text.
# One item may belong to multiple genres.
# ============================================================

GENRE_KEYWORDS = {
    "Romance": [
        "romance",
        "romantic",
        "love story",
        "love stories",
        "relationship",
        "relationships",
        "fall in love",
    ],

    "Fantasy": [
        "fantasy",
        "magic",
        "magical",
        "mythology",
        "mythological",
        "wizard",
        "witch",
        "dragon",
        "supernatural",
        "paranormal",
    ],

    "Mystery / Thriller": [
        "mystery",
        "mysteries",
        "thriller",
        "suspense",
        "crime",
        "detective",
        "murder",
        "investigation",
    ],

    "Science Fiction": [
        "science fiction",
        "sci-fi",
        "scifi",
        "space opera",
        "dystopian",
        "futuristic",
        "alien",
        "aliens",
    ],

    "Historical": [
        "historical fiction",
        "historical novel",
        "historical romance",
        "historical",
        "history lovers",
        "period novel",
    ],

    "Young Adult": [
        "young adult",
        "young readers",
        "teen reader",
        "teen readers",
        "teenage readers",
        "ya novel",
        "ya fantasy",
    ],

    "Children": [
        "children's book",
        "children book",
        "young children",
        "kids",
        "child readers",
        "picture book",
        "middle grade",
    ],

    "Horror": [
        "horror",
        "scary",
        "frightening",
        "terrifying",
        "ghost story",
        "haunted",
    ],

    "Biography / Memoir": [
        "biography",
        "biographical",
        "memoir",
        "autobiography",
        "life story",
    ],

    "Self-Help": [
        "self-help",
        "self help",
        "personal development",
        "self-improvement",
        "self improvement",
        "motivational",
    ],

    "Religion / Spirituality": [
        "religion",
        "religious",
        "christian",
        "christianity",
        "bible",
        "biblical",
        "spiritual",
        "spirituality",
        "faith",
    ],
}


# ============================================================
# Genre tagging
# ============================================================

def extract_genres(profile_dict):
    """
    Infer one or more genre labels from profile + reasoning.

    These are heuristic proxy genre labels rather than
    ground-truth Amazon genre metadata.
    """

    profile = str(profile_dict.get("profile", ""))
    reasoning = str(profile_dict.get("reasoning", ""))

    text = (profile + " " + reasoning).lower()

    genres = []

    for genre, keywords in GENRE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            genres.append(genre)

    if not genres:
        genres.append("Other / Unclassified")

    return genres


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Genre-level Gate Comparison")
    print("CLIP Text + CLIP Image vs OpenAI Text + CLIP Image")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load profiles
    # --------------------------------------------------------

    print("\n[1] Loading item profiles...")

    with open(ITEM_PROFILE_PATH, "rb") as f:
        item_profiles = pickle.load(f)

    print("Items:", len(item_profiles))

    # --------------------------------------------------------
    # 2. Load gates
    # --------------------------------------------------------

    print("\n[2] Loading gate values...")

    clip_gates = np.load(CLIP_GATE_PATH).reshape(-1)
    openai_gates = np.load(OPENAI_GATE_PATH).reshape(-1)

    print("CLIP gates  :", clip_gates.shape)
    print("OpenAI gates:", openai_gates.shape)

    # --------------------------------------------------------
    # 3. Alignment checks
    # --------------------------------------------------------

    assert len(item_profiles) == len(clip_gates), (
        "CLIP gate count does not match item profiles."
    )

    assert len(item_profiles) == len(openai_gates), (
        "OpenAI gate count does not match item profiles."
    )

    assert list(item_profiles.keys()) == list(
        range(len(item_profiles))
    ), "Item profile IDs are not aligned with Amazon7425 item IDs."

    print("PASS: Item profiles and both gate arrays are aligned.")

    # --------------------------------------------------------
    # 4. Genre tagging
    # --------------------------------------------------------

    print("\n[3] Extracting genre tags...")

    genre_records = defaultdict(list)

    item_rows = []

    unclassified_count = 0
    multi_label_count = 0

    for item_id in range(len(item_profiles)):

        genres = extract_genres(item_profiles[item_id])

        if genres == ["Other / Unclassified"]:
            unclassified_count += 1

        if len(genres) > 1:
            multi_label_count += 1

        clip_gate = float(clip_gates[item_id])
        openai_gate = float(openai_gates[item_id])

        for genre in genres:

            record = {
                "item_id": item_id,
                "genre": genre,
                "clip_gate": clip_gate,
                "openai_gate": openai_gate,
                "clip_text_dominant": clip_gate > 0.5,
                "openai_text_dominant": openai_gate > 0.5,
            }

            genre_records[genre].append(record)
            item_rows.append(record)

    print("Total tagged records :", len(item_rows))
    print("Multi-label items    :", multi_label_count)
    print("Unclassified items   :", unclassified_count)

    # --------------------------------------------------------
    # 5. Save item-level tags
    # --------------------------------------------------------

    with open(
        ITEM_OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "item_id",
            "genre",
            "clip_gate",
            "openai_gate",
            "clip_text_dominant",
            "openai_text_dominant",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(item_rows)

    print("\nItem-level tags saved:")
    print(ITEM_OUTPUT_PATH)

    # --------------------------------------------------------
    # 6. Genre statistics
    # --------------------------------------------------------

    print("\n[4] Computing genre-level statistics...")

    summaries = []

    for genre, records in genre_records.items():

        clip_values = np.array(
            [r["clip_gate"] for r in records],
            dtype=float
        )

        openai_values = np.array(
            [r["openai_gate"] for r in records],
            dtype=float
        )

        n = len(records)

        clip_mean = float(np.mean(clip_values))
        openai_mean = float(np.mean(openai_values))

        clip_median = float(np.median(clip_values))
        openai_median = float(np.median(openai_values))

        clip_std = float(np.std(clip_values))
        openai_std = float(np.std(openai_values))

        clip_text_pct = float(
            np.mean(clip_values > 0.5) * 100
        )

        openai_text_pct = float(
            np.mean(openai_values > 0.5) * 100
        )

        summaries.append({
            "genre": genre,
            "items": n,

            "clip_mean_gate": clip_mean,
            "clip_median_gate": clip_median,
            "clip_std": clip_std,

            "openai_mean_gate": openai_mean,
            "openai_median_gate": openai_median,
            "openai_std": openai_std,

            "gate_shift_openai_minus_clip":
                openai_mean - clip_mean,

            "clip_text_dominant_pct":
                clip_text_pct,

            "openai_text_dominant_pct":
                openai_text_pct,
        })

    summaries.sort(
        key=lambda x: x["items"],
        reverse=True
    )

    # --------------------------------------------------------
    # 7. Save summary CSV
    # --------------------------------------------------------

    summary_fields = [
        "genre",
        "items",

        "clip_mean_gate",
        "clip_median_gate",
        "clip_std",

        "openai_mean_gate",
        "openai_median_gate",
        "openai_std",

        "gate_shift_openai_minus_clip",

        "clip_text_dominant_pct",
        "openai_text_dominant_pct",
    ]

    with open(
        SUMMARY_OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields
        )

        writer.writeheader()
        writer.writerows(summaries)

    # --------------------------------------------------------
    # 8. Print genre results
    # --------------------------------------------------------

    print("\n" + "=" * 95)
    print("Genre-level Gate Results")
    print("=" * 95)

    header = (
        f"{'Genre':<25}"
        f"{'N':>8}"
        f"{'CLIP':>12}"
        f"{'OpenAI':>12}"
        f"{'Shift':>12}"
        f"{'CLIP T%':>12}"
        f"{'OpenAI T%':>12}"
    )

    print(header)
    print("-" * 95)

    for row in summaries:

        print(
            f"{row['genre']:<25}"
            f"{row['items']:>8}"
            f"{row['clip_mean_gate']:>12.4f}"
            f"{row['openai_mean_gate']:>12.4f}"
            f"{row['gate_shift_openai_minus_clip']:>+12.4f}"
            f"{row['clip_text_dominant_pct']:>11.2f}%"
            f"{row['openai_text_dominant_pct']:>11.2f}%"
        )

    # --------------------------------------------------------
    # 9. Main genres for plotting
    # --------------------------------------------------------

    MIN_ITEMS = 30

    plot_rows = [
        row for row in summaries
        if row["items"] >= MIN_ITEMS
        and row["genre"] != "Other / Unclassified"
    ]

    print(
        f"\nGenres with >= {MIN_ITEMS} items "
        f"(excluding Other / Unclassified): "
        f"{len(plot_rows)}"
    )

    # --------------------------------------------------------
    # 10. Comparison plot
    # --------------------------------------------------------

    if plot_rows:

        plot_rows.sort(
            key=lambda x: x["openai_mean_gate"]
        )

        genres = [
            row["genre"]
            for row in plot_rows
        ]

        clip_means = [
            row["clip_mean_gate"]
            for row in plot_rows
        ]

        openai_means = [
            row["openai_mean_gate"]
            for row in plot_rows
        ]

        x = np.arange(len(genres))
        width = 0.36

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.bar(
            x - width / 2,
            clip_means,
            width,
            label="CLIP Text + CLIP Image"
        )

        ax.bar(
            x + width / 2,
            openai_means,
            width,
            label="OpenAI Text + CLIP Image"
        )

        ax.axhline(
            0.5,
            linestyle="--",
            linewidth=1
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            genres,
            rotation=35,
            ha="right"
        )

        ax.set_ylabel(
            "Mean Gate Value (Text Weight)"
        )

        ax.set_xlabel(
            "Inferred Genre"
        )

        ax.set_title(
            "Learned Text-Modality Weight by Genre"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            FIGURE_OUTPUT_PATH,
            dpi=300
        )

        plt.close(fig)

        print("\nFigure saved:")
        print(FIGURE_OUTPUT_PATH)

    # --------------------------------------------------------
    # 11. Overall comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Overall Gate Comparison")
    print("=" * 70)

    clip_mean = float(np.mean(clip_gates))
    openai_mean = float(np.mean(openai_gates))

    print(
        f"CLIP mean gate        : "
        f"{clip_mean:.6f}"
    )

    print(
        f"OpenAI mean gate      : "
        f"{openai_mean:.6f}"
    )

    print(
        f"Mean gate shift       : "
        f"{openai_mean - clip_mean:+.6f}"
    )

    print(
        f"CLIP text-dominant    : "
        f"{np.mean(clip_gates > 0.5) * 100:.2f}%"
    )

    print(
        f"OpenAI text-dominant  : "
        f"{np.mean(openai_gates > 0.5) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 12. Genre range
    # --------------------------------------------------------

    main_genres = [
        row for row in summaries
        if row["items"] >= MIN_ITEMS
        and row["genre"] != "Other / Unclassified"
    ]

    if main_genres:

        clip_genre_means = [
            row["clip_mean_gate"]
            for row in main_genres
        ]

        openai_genre_means = [
            row["openai_mean_gate"]
            for row in main_genres
        ]

        shifts = [
            row["gate_shift_openai_minus_clip"]
            for row in main_genres
        ]

        print("\n" + "=" * 70)
        print("Main-Genre Range Analysis")
        print("=" * 70)

        print(
            f"CLIP genre mean range   : "
            f"{min(clip_genre_means):.4f} - "
            f"{max(clip_genre_means):.4f}"
        )

        print(
            f"OpenAI genre mean range : "
            f"{min(openai_genre_means):.4f} - "
            f"{max(openai_genre_means):.4f}"
        )

        print(
            f"Gate shift range        : "
            f"{min(shifts):+.4f} - "
            f"{max(shifts):+.4f}"
        )

    print("\nSummary CSV saved:")
    print(SUMMARY_OUTPUT_PATH)

    print("\nGenre analysis completed.")


if __name__ == "__main__":
    main()