import torch

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model

print("=" * 60)
print("RLMRec Multimodal Sanity Check")
print("=" * 60)

##########################################################
# Build Dataset
##########################################################

print("\n[1] Loading dataset...")

data_handler = build_data_handler()
data_handler.load_data()

print("Done.")

##########################################################
# Build Model
##########################################################

print("\n[2] Building model...")

model = build_model(data_handler).to(configs["device"])

print("Done.")

##########################################################
# Load Checkpoint
##########################################################

print("\n[3] Loading checkpoint...")

ckpt_path = "./checkpoint/lightgcn_plus/lightgcn_plus-amazon-2023.pth"

checkpoint = torch.load(
    ckpt_path,
    map_location=configs["device"]
)

model.load_state_dict(checkpoint)

model.eval()

print("Checkpoint loaded!")

##########################################################
# Raw Feature Shapes
##########################################################

print("\n[4] Raw Feature Shapes")

print("User profile:", model.usrprf_embeds.shape)
print("Item text   :", model.item_text_embeds.shape)
print("Item image  :", model.item_image_embeds.shape)

##########################################################
# Projection
##########################################################

print("\n[5] Projection")

with torch.no_grad():

    user = model.user_projection(
        model.usrprf_embeds[:5]
    )

    text = model.text_projection(
        model.item_text_embeds[:5]
    )

    image = model.image_projection(
        model.item_image_embeds[:5]
    )

print("User :", user.shape)
print("Text :", text.shape)
print("Image:", image.shape)

##########################################################
# Fusion
##########################################################

print("\n[6] Fusion")

with torch.no_grad():

    fused = model.fusion(
        text,
        image
    )

print("Fusion:", fused.shape)

##########################################################
# Norm Check
##########################################################

print("\n[7] Feature Norm")

print("User :", torch.norm(user).item())
print("Text :", torch.norm(text).item())
print("Image:", torch.norm(image).item())
print("Fusion:", torch.norm(fused).item())

##########################################################
# Image Contribution Test
##########################################################

print("\n[8] Image Contribution")

with torch.no_grad():

    zero_image = torch.zeros_like(image)

    fused_without_image = model.fusion(
        text,
        zero_image
    )

difference = torch.norm(
    fused - fused_without_image
).item()

print("Difference =", difference)

if difference < 1e-6:
    print("WARNING: Image branch contributes almost nothing!")
else:
    print("PASS: Image branch contributes.")

##########################################################
# Parameter Norm
##########################################################

print("\n[9] Trainable Module Norm")

print(
    "User Projection :",
    model.user_projection.network[0].weight.norm().item()
)

print(
    "Text Projection :",
    model.text_projection.network[0].weight.norm().item()
)

print(
    "Image Projection:",
    model.image_projection.network[0].weight.norm().item()
)

print(
    "Fusion          :",
    model.fusion.network[0].weight.norm().item()
)

##########################################################
# Finish
##########################################################

print("\n" + "=" * 60)
print("Sanity Check Finished!")
print("=" * 60)