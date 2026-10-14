import csv
import json
import pickle
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

ANALYSIS_DIR = Path("generation/analysis")

GATES_BY_SEED_PATH = ANALYSIS_DIR / (
    "gated_openai_clip_image_amazon7425_multiseed_gates_by_seed.npy"
)

GATE_MEAN_PATH = ANALYSIS_DIR / (
    "gated_openai_clip_image_amazon7425_multiseed_gate_mean.npy"
)

GATE_STD_PATH = ANALYSIS_DIR / (
    "gated_openai_clip_image_amazon7425_multiseed_gate_std.npy"
)

IMAGE_MAPPING_PATH = Path("iid_to_image.json")

ITEM_PROFILE_PATH = Path(
    "data/amazon7425/itm_prf.pkl"
)

OUTPUT_DIR = ANALYSIS_DIR / (
    "extreme_image_weights_multiseed"
)

HIGH_IMAGE_DIR = OUTPUT_DIR / "high_image_weight"
LOW_IMAGE_DIR = OUTPUT_DIR / "low_image_weight"

TOP_K = 50
SEEDS = [2023, 2024, 2025]

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# ============================================================
# Setup
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HIGH_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LOW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load gate data
# ============================================================

print("=" * 70)
print("Multi-seed Extreme Image Weight Analysis")
print("=" * 70)

print("\nLoading multi-seed gate values...")

gates_by_seed = np.load(
    GATES_BY_SEED_PATH
)

gate_mean = np.load(
    GATE_MEAN_PATH
).reshape(-1)

gate_std = np.load(
    GATE_STD_PATH
).reshape(-1)

print("Gates-by-seed shape:", gates_by_seed.shape)
print("Gate mean shape     :", gate_mean.shape)
print("Gate std shape      :", gate_std.shape)

assert gates_by_seed.shape == (3, 7425)
assert len(gate_mean) == 7425
assert len(gate_std) == 7425

image_weights_by_seed = 1.0 - gates_by_seed
image_weight_mean = 1.0 - gate_mean

# Because w_image = 1 - gate, std is identical.
image_weight_std = gate_std

print("\nMean gate statistics")
print("Mean:", gate_mean.mean())
print("Min :", gate_mean.min())
print("Max :", gate_mean.max())

print("\nMean image-weight statistics")
print("Mean:", image_weight_mean.mean())
print("Min :", image_weight_mean.min())
print("Max :", image_weight_mean.max())


# ============================================================
# Load mappings / profiles
# ============================================================

print("\nLoading image mapping...")

with open(
    IMAGE_MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:
    iid_to_image = json.load(f)

keep_items = sorted(
    int(i)
    for i in iid_to_image.keys()
)

print("Items with image:", len(keep_items))

assert len(keep_items) == 7425


print("\nLoading item profiles...")

with open(
    ITEM_PROFILE_PATH,
    "rb"
) as f:
    item_profiles = pickle.load(f)

assert len(item_profiles) == 7425


# ============================================================
# Build complete item table
# ============================================================

items = []

for new_id in range(7425):

    old_id = keep_items[new_id]

    profile_data = item_profiles[new_id]

    if isinstance(profile_data, dict):
        profile = profile_data.get("profile", "")
        reasoning = profile_data.get("reasoning", "")
    else:
        profile = str(profile_data)
        reasoning = ""

    item = {
        "new_id": new_id,
        "old_id": old_id,

        "gate_2023": float(
            gates_by_seed[0, new_id]
        ),
        "gate_2024": float(
            gates_by_seed[1, new_id]
        ),
        "gate_2025": float(
            gates_by_seed[2, new_id]
        ),

        "gate_mean": float(
            gate_mean[new_id]
        ),
        "gate_std": float(
            gate_std[new_id]
        ),

        "image_weight_2023": float(
            image_weights_by_seed[0, new_id]
        ),
        "image_weight_2024": float(
            image_weights_by_seed[1, new_id]
        ),
        "image_weight_2025": float(
            image_weights_by_seed[2, new_id]
        ),

        "image_weight_mean": float(
            image_weight_mean[new_id]
        ),
        "image_weight_std": float(
            image_weight_std[new_id]
        ),

        "image_url": iid_to_image[str(old_id)],
        "profile": profile,
        "reasoning": reasoning,
    }

    items.append(item)


# ============================================================
# Select extremes using THREE-SEED MEAN
# ============================================================

# Largest average image weight
high_image_items = sorted(
    items,
    key=lambda x: x["image_weight_mean"],
    reverse=True
)[:TOP_K]

# Smallest average image weight
low_image_items = sorted(
    items,
    key=lambda x: x["image_weight_mean"]
)[:TOP_K]


# ============================================================
# Print top examples
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 LARGEST MEAN IMAGE WEIGHTS")
print("=" * 70)

for item in high_image_items[:10]:

    print(
        f"new_id={item['new_id']:4d} | "
        f"old_id={item['old_id']:4d} | "
        f"image_mean={item['image_weight_mean']:.6f} | "
        f"std={item['image_weight_std']:.6f}"
    )


print("\n" + "=" * 70)
print("TOP 10 SMALLEST MEAN IMAGE WEIGHTS")
print("=" * 70)

for item in low_image_items[:10]:

    print(
        f"new_id={item['new_id']:4d} | "
        f"old_id={item['old_id']:4d} | "
        f"image_mean={item['image_weight_mean']:.6f} | "
        f"std={item['image_weight_std']:.6f}"
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(items, path):

    fieldnames = [
        "rank",
        "amazon7425_id",
        "original_amazon_id",

        "gate_2023",
        "gate_2024",
        "gate_2025",
        "gate_mean",
        "gate_std",

        "image_weight_2023",
        "image_weight_2024",
        "image_weight_2025",
        "image_weight_mean",
        "image_weight_std",

        "image_url",
        "profile",
        "reasoning",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for rank, item in enumerate(
            items,
            start=1
        ):

            row = {
                "rank": rank,
                "amazon7425_id": item["new_id"],
                "original_amazon_id": item["old_id"],

                "gate_2023": item["gate_2023"],
                "gate_2024": item["gate_2024"],
                "gate_2025": item["gate_2025"],
                "gate_mean": item["gate_mean"],
                "gate_std": item["gate_std"],

                "image_weight_2023":
                    item["image_weight_2023"],
                "image_weight_2024":
                    item["image_weight_2024"],
                "image_weight_2025":
                    item["image_weight_2025"],

                "image_weight_mean":
                    item["image_weight_mean"],

                "image_weight_std":
                    item["image_weight_std"],

                "image_url": item["image_url"],
                "profile": item["profile"],
                "reasoning": item["reasoning"],
            }

            writer.writerow(row)


high_csv = (
    OUTPUT_DIR /
    "top50_highest_mean_image_weights.csv"
)

low_csv = (
    OUTPUT_DIR /
    "top50_lowest_mean_image_weights.csv"
)

save_csv(
    high_image_items,
    high_csv
)

save_csv(
    low_image_items,
    low_csv
)

print("\nSaved:")
print(high_csv)
print(low_csv)


# ============================================================
# Download / collect covers
# ============================================================

def download_images(items, output_dir):

    print(
        f"\nCollecting images into {output_dir}..."
    )

    success = 0
    failed = 0

    for rank, item in enumerate(
        tqdm(items),
        start=1
    ):

        new_id = item["new_id"]
        old_id = item["old_id"]

        filename = (
            f"{rank:02d}_"
            f"new{new_id}_"
            f"old{old_id}_"
            f"w{item['image_weight_mean']:.4f}_"
            f"std{item['image_weight_std']:.4f}.jpg"
        )

        save_path = (
            output_dir /
            filename
        )

        item["local_path"] = str(save_path)

        if save_path.exists():

            try:
                with Image.open(save_path) as img:
                    img.verify()

                success += 1
                continue

            except Exception:
                save_path.unlink()

        try:

            response = requests.get(
                item["image_url"],
                headers=HEADERS,
                timeout=20
            )

            if response.status_code != 200:

                print(
                    f"\nHTTP "
                    f"{response.status_code}: "
                    f"old_id={old_id}"
                )

                failed += 1
                continue

            with open(
                save_path,
                "wb"
            ) as f:

                f.write(
                    response.content
                )

            with Image.open(save_path) as img:
                img.verify()

            success += 1

        except Exception as e:

            print(
                f"\nFailed new_id={new_id}, "
                f"old_id={old_id}: {e}"
            )

            failed += 1

    print(
        "Success:",
        success
    )

    print(
        "Failed :",
        failed
    )


download_images(
    high_image_items,
    HIGH_IMAGE_DIR
)

download_images(
    low_image_items,
    LOW_IMAGE_DIR
)


print("\n" + "=" * 70)
print("Multi-seed extreme image-weight analysis completed.")
print("=" * 70)