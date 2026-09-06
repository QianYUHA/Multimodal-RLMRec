from config.configurator import configs

print("=" * 60)
print("E2 CLIP Text Config Check")
print("=" * 60)

print("Dataset:", configs["data"]["name"])
print("Model:", configs["model"]["name"])

print("User OpenAI:", configs["usrprf_embeds"].shape)
print("Item OpenAI:", configs["itmprf_embeds"].shape)

if "item_text_embeds" in configs:
    print("Item CLIP Text:", configs["item_text_embeds"].shape)

if "item_image_embeds" in configs:
    print("Item CLIP Image:", configs["item_image_embeds"].shape)