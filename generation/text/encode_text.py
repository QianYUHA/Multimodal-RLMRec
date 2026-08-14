import pickle
from pathlib import Path

import clip
import torch
import numpy as np
from tqdm import tqdm

# -------------------------
# Config
# -------------------------

INPUT_PATH = Path("data/amazon/itm_prf.pkl")
OUTPUT_PATH = Path("generation/features/item_text_clip.pkl")

NUM_ITEMS = 9332
EMBED_DIM = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP...")

model, _ = clip.load("ViT-B/32", device=DEVICE)
model.eval()

with open(INPUT_PATH, "rb") as f:
    item_profiles = pickle.load(f)

text_features = np.zeros(
    (NUM_ITEMS, EMBED_DIM),
    dtype=np.float32
)

with torch.no_grad():

    for iid in tqdm(range(NUM_ITEMS)):

        profile = item_profiles[iid]["profile"]

        text = clip.tokenize(
            [profile],
            truncate=True
        ).to(DEVICE)

        text_feature = model.encode_text(text)

        text_feature = text_feature / text_feature.norm(
            dim=-1,
            keepdim=True
        )

        text_features[iid] = (
            text_feature
            .squeeze(0)
            .cpu()
            .numpy()
        )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(text_features, f)

print("Saved:", OUTPUT_PATH)
print(text_features.shape)

