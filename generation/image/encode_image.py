import os
import pickle
from pathlib import Path

import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==========================
# Config
# ==========================

IMAGE_DIR = Path("data/book_images")
OUTPUT_PATH = Path("generation/features/item_image_clip.pkl")

NUM_ITEMS = 9332
EMBED_DIM = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Generate CLIP Image Embeddings")
print(f"Device : {DEVICE}")
print("=" * 60)

# -------------------------
# Load CLIP
# -------------------------
print("Loading CLIP...")

model, preprocess = clip.load(
    "ViT-B/32",
    device=DEVICE
)

model.eval()

# -------------------------
# Encode
# -------------------------

image_features = np.zeros(
    (NUM_ITEMS, EMBED_DIM),
    dtype=np.float32
)

image_paths = sorted(IMAGE_DIR.glob("*"))

print(f"Found {len(image_paths)} images.")

# -------------------------
# Encode Images
# -------------------------

print("Encoding images...")

with torch.no_grad():

    for image_path in tqdm(image_paths):

        try:

            # Read image
            image = Image.open(image_path).convert("RGB")

            # CLIP preprocess
            image_input = preprocess(image).unsqueeze(0).to(DEVICE)

            # Encode
            image_feature = model.encode_image(image_input)

            # L2 Normalize
            image_feature = image_feature / image_feature.norm(
                dim=-1,
                keepdim=True
            )

            # GPU -> CPU -> numpy
            image_feature = image_feature.squeeze(0).cpu().numpy()

            # image filename -> iid
            iid = int(image_path.stem)

            image_features[iid] = image_feature

        except Exception as e:

            print(f"Error: {image_path.name}")
            print(e)

# -------------------------
# Save Features
# -------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(image_features, f)

print("=" * 60)
print("Finished!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Feature Shape: {image_features.shape}")
print("=" * 60)