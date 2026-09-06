from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model

import torch


print("=" * 60)
print("E2 CLIP Text Model Check")
print("=" * 60)

print("Dataset:", configs["data"]["name"])
print("Model:", configs["model"]["name"])

# -------------------------
# Load data
# -------------------------

data_handler = build_data_handler()
data_handler.load_data()

print("\nData:")
print("Users:", configs["data"]["user_num"])
print("Items:", configs["data"]["item_num"])

# -------------------------
# Build model
# -------------------------

model = build_model(data_handler)
model = model.to(configs["device"])

print("\nModel:")
print(type(model).__name__)

# -------------------------
# Check embeddings
# -------------------------

print("\nSemantic embeddings:")

print(
    "User OpenAI:",
    model.usrprf_embeds.shape
)

print(
    "Item CLIP Text:",
    model.item_text_embeds.shape
)

# -------------------------
# Projection test
# -------------------------

with torch.no_grad():

    user_semantic = model.user_mlp(
        model.usrprf_embeds[:4]
    )

    item_text_semantic = model.text_projection(
        model.item_text_embeds[:4]
    )

print("\nProjected embeddings:")

print(
    "User:",
    user_semantic.shape
)

print(
    "Item:",
    item_text_semantic.shape
)

# -------------------------
# Assertions
# -------------------------

assert model.usrprf_embeds.shape == (10989, 1536)

assert model.item_text_embeds.shape == (7425, 512)

assert user_semantic.shape == (4, 32)

assert item_text_semantic.shape == (4, 32)

print("\n" + "=" * 60)
print("PASS: E2 model initialization and projection are correct.")
print("=" * 60)