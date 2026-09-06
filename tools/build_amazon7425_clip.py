import pickle
import numpy as np
import os

# ============================
# Paths
# ============================

ORIGINAL_CLIP_TEXT = (
    "generation/features/item_text_clip.pkl"
)

OUTPUT_DIR = "data/amazon7425"
OUTPUT_PATH = f"{OUTPUT_DIR}/item_text_clip.pkl"

IMAGE_MAPPING = "iid_to_image.json"


# ============================
# Load CLIP text embeddings
# ============================

with open(ORIGINAL_CLIP_TEXT, "rb") as f:
    clip_text = pickle.load(f)

print("=" * 60)
print("Original CLIP Text Embedding")
print("=" * 60)

print("Type:", type(clip_text))
print("Shape:", clip_text.shape)


# ============================
# Load image mapping
# ============================

import json

with open(IMAGE_MAPPING, "r") as f:
    iid_to_image = json.load(f)

keep_items = sorted(
    [int(i) for i in iid_to_image.keys()]
)

print("\nItems with image:", len(keep_items))


# ============================
# Filter
# ============================

new_clip_text = clip_text[keep_items]

print("\nFiltered CLIP Text")
print("Shape:", new_clip_text.shape)


# ============================
# Sanity check
# ============================

assert clip_text.shape == (9332, 512)

assert len(keep_items) == 7425

assert new_clip_text.shape == (7425, 512)

print("\nPASS: CLIP text embeddings aligned.")


# ============================
# Save
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(new_clip_text, f)

print("\nSaved to:")
print(OUTPUT_PATH)