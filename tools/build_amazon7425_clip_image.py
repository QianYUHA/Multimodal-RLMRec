import pickle
import numpy as np
import json
import os

ORIGINAL_IMAGE = "generation/features/item_image_clip.pkl"

OUTPUT_DIR = "data/amazon7425"
OUTPUT_PATH = f"{OUTPUT_DIR}/item_image_clip.pkl"

IMAGE_MAPPING = "iid_to_image.json"


# ============================
# Load original image embedding
# ============================

with open(ORIGINAL_IMAGE, "rb") as f:
    clip_image = pickle.load(f)

print("=" * 60)
print("Original CLIP Image Embedding")
print("=" * 60)

print("Type:", type(clip_image))
print("Shape:", clip_image.shape)


# ============================
# Load item mapping
# ============================

with open(IMAGE_MAPPING, "r") as f:
    iid_to_image = json.load(f)

keep_items = sorted(
    [int(i) for i in iid_to_image.keys()]
)

print("\nItems with image:", len(keep_items))


# ============================
# Filter
# ============================

new_clip_image = clip_image[keep_items]

print("\nFiltered CLIP Image")
print("Shape:", new_clip_image.shape)


# ============================
# Sanity check
# ============================

assert clip_image.shape == (9332, 512)

assert len(keep_items) == 7425

assert new_clip_image.shape == (7425, 512)

print("\nPASS: CLIP image embeddings aligned.")


# ============================
# Save
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(new_clip_image, f)

print("\nSaved to:")
print(OUTPUT_PATH)