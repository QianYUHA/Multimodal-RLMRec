import pickle
import numpy as np
import csv
import os


# ============================================================
# Paths
# ============================================================

GATE_PATH = "generation/analysis/gated_openai_clip_image_amazon7425_gate.npy"
ITEM_PROFILE_PATH = "data/amazon7425/itm_prf.pkl"

OUTPUT_PATH = "generation/analysis/gated_openai_clip_image_amazon7425_items.csv"


# ============================================================
# Load gate values
# ============================================================

gate_values = np.load(GATE_PATH).reshape(-1)

print("Gate shape:", gate_values.shape)


# ============================================================
# Load item profiles
# ============================================================

with open(ITEM_PROFILE_PATH, "rb") as f:
    item_profiles = pickle.load(f)

print("Item profiles:", len(item_profiles))


# ============================================================
# Sanity check
# ============================================================

assert len(gate_values) == len(item_profiles), \
    "Gate values and item profiles have different lengths."


assert list(item_profiles.keys()) == list(range(len(item_profiles))), \
    "Item profile keys are not aligned with Amazon7425 item IDs."


print("PASS: Gate and item profiles are aligned.")


# ============================================================
# Build records
# ============================================================

records = []

for item_id in range(len(gate_values)):

    gate = float(gate_values[item_id])

    text_weight = gate
    image_weight = 1.0 - gate

    if gate > 0.5:
        modality = "text-dominant"
    elif gate < 0.5:
        modality = "image-dominant"
    else:
        modality = "balanced"

    profile = item_profiles[item_id]

    records.append({
        "item_id": item_id,
        "gate": gate,
        "text_weight": text_weight,
        "image_weight": image_weight,
        "modality": modality,
        "profile": profile.get("profile", ""),
        "reasoning": profile.get("reasoning", "")
    })


# ============================================================
# Sort by gate
# ============================================================

records.sort(key=lambda x: x["gate"])


# ============================================================
# Save CSV
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "item_id",
            "gate",
            "text_weight",
            "image_weight",
            "modality",
            "profile",
            "reasoning"
        ]
    )

    writer.writeheader()
    writer.writerows(records)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 60)
print("Gate Analysis Export")
print("=" * 60)

print("Total items:", len(records))

print()
print("Image-dominant items:", sum(
    x["modality"] == "image-dominant"
    for x in records
))

print("Text-dominant items:", sum(
    x["modality"] == "text-dominant"
    for x in records
))

print()
print("Lowest 10 gate values")
print("-" * 60)

for x in records[:10]:
    print(
        f"item={x['item_id']:4d} "
        f"gate={x['gate']:.6f} "
        f"image_weight={x['image_weight']:.6f}"
    )

print()
print("Highest 10 gate values")
print("-" * 60)

for x in records[-10:][::-1]:
    print(
        f"item={x['item_id']:4d} "
        f"gate={x['gate']:.6f} "
        f"text_weight={x['text_weight']:.6f}"
    )

print()
print("Saved to:")
print(OUTPUT_PATH)