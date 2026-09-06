import json
import pickle
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

IMAGE_DIR = Path("data/book_images")
MAPPING_PATH = Path("iid_to_image.json")

OUTPUT_PATH = Path(
    "generation/features/item_image_clip.pkl"
)

NUM_ITEMS = 9332
EMBED_DIM = 512

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 128


# ============================================================
# Load expected image IDs
# ============================================================

print("=" * 70)
print("Generate Complete CLIP Image Embeddings")
print("=" * 70)

with open(
    MAPPING_PATH,
    "r",
    encoding="utf-8",
) as f:
    mapping = json.load(f)

expected_ids = sorted(
    int(iid)
    for iid in mapping.keys()
)

print("Expected image items:", len(expected_ids))

assert len(expected_ids) == 7425, (
    f"Expected 7425 mapped items, "
    f"found {len(expected_ids)}"
)


# ============================================================
# Validate image files BEFORE loading CLIP
# ============================================================

print("\nValidating image files...")

missing = []
invalid = []

for iid in tqdm(expected_ids):

    image_path = (
        IMAGE_DIR / f"{iid}.jpg"
    )

    if not image_path.exists():
        missing.append(iid)
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()

    except Exception:
        invalid.append(iid)


print("\nImage validation:")
print("Expected:", len(expected_ids))
print("Missing :", len(missing))
print("Invalid :", len(invalid))

if missing:
    print("First missing IDs:")
    print(missing[:20])

if invalid:
    print("First invalid IDs:")
    print(invalid[:20])

if missing or invalid:
    raise RuntimeError(
        "Image dataset is incomplete. "
        "Embedding generation aborted."
    )

print(
    "PASS: All 7425 mapped images exist "
    "and are readable."
)


# ============================================================
# Load CLIP
# ============================================================

print("\nLoading CLIP ViT-B/32...")
print("Device:", DEVICE)

model, preprocess = clip.load(
    "ViT-B/32",
    device=DEVICE,
)

model.eval()


# ============================================================
# Allocate ORIGINAL 9332-item feature matrix
#
# Items without an image remain zero.
# All 7425 mapped items MUST become non-zero.
# ============================================================

image_features = np.zeros(
    (NUM_ITEMS, EMBED_DIM),
    dtype=np.float32,
)


# ============================================================
# Batch encoding
# ============================================================

print("\nEncoding images...")

with torch.no_grad():

    for start in tqdm(
        range(0, len(expected_ids), BATCH_SIZE),
        desc="CLIP encoding",
    ):

        batch_ids = expected_ids[
            start:start + BATCH_SIZE
        ]

        batch_images = []

        for iid in batch_ids:

            image_path = (
                IMAGE_DIR / f"{iid}.jpg"
            )

            try:
                image = Image.open(
                    image_path
                ).convert("RGB")

                image_input = preprocess(
                    image
                )

                batch_images.append(
                    image_input
                )

            except Exception as e:
                raise RuntimeError(
                    f"Failed reading IID {iid}: {e}"
                )

        batch_tensor = torch.stack(
            batch_images
        ).to(DEVICE)

        batch_features = model.encode_image(
            batch_tensor
        )

        # L2 normalize
        batch_features = (
            batch_features
            / batch_features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        batch_features = (
            batch_features
            .float()
            .cpu()
            .numpy()
        )

        for iid, feature in zip(
            batch_ids,
            batch_features,
        ):
            image_features[iid] = feature


# ============================================================
# Critical sanity checks
# ============================================================

print("\n" + "=" * 70)
print("Embedding Sanity Check")
print("=" * 70)

mapped_features = image_features[
    expected_ids
]

norms = np.linalg.norm(
    mapped_features,
    axis=1,
)

zero_mask = norms < 1e-8

zero_count = int(
    zero_mask.sum()
)

print(
    "Full feature shape :",
    image_features.shape,
)

print(
    "Mapped shape       :",
    mapped_features.shape,
)

print(
    "Mean mapped norm   :",
    norms.mean(),
)

print(
    "Min mapped norm    :",
    norms.min(),
)

print(
    "Max mapped norm    :",
    norms.max(),
)

print(
    "Zero mapped vectors:",
    zero_count,
)

print(
    "Near-unit vectors  :",
    np.sum(
        np.abs(norms - 1.0) < 1e-3
    ),
)


# ============================================================
# FAIL FAST
# ============================================================

if zero_count != 0:

    bad_ids = np.array(
        expected_ids
    )[zero_mask]

    print(
        "Zero-vector item IDs:"
    )
    print(
        bad_ids[:20]
    )

    raise RuntimeError(
        f"{zero_count} mapped image embeddings "
        "are zero. Output will NOT be saved."
    )


if not np.all(
    np.isfinite(mapped_features)
):

    raise RuntimeError(
        "NaN or Inf detected in image "
        "embeddings. Output will NOT be saved."
    )


if mapped_features.shape != (
    7425,
    512,
):

    raise RuntimeError(
        "Unexpected mapped embedding shape."
    )


print(
    "\nPASS: All 7425 mapped items "
    "have valid non-zero CLIP embeddings."
)


# ============================================================
# Save ONLY after every check passes
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_PATH,
    "wb",
) as f:

    pickle.dump(
        image_features,
        f,
    )


print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_PATH,
)

print(
    "Original feature shape:",
    image_features.shape,
)

print(
    "Mapped valid embeddings:",
    len(expected_ids),
)

print("=" * 70)