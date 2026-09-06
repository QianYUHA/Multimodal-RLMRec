import csv
import json
import pickle
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

GATE_PATH = Path(
    "generation/analysis/"
    "gated_openai_clip_image_amazon7425_gate.npy"
)

IMAGE_MAPPING_PATH = Path("iid_to_image.json")

ITEM_PROFILE_PATH = Path(
    "data/amazon7425/itm_prf.pkl"
)

OUTPUT_DIR = Path(
    "generation/analysis/extreme_image_weights"
)

HIGH_IMAGE_DIR = OUTPUT_DIR / "high_image_weight"
LOW_IMAGE_DIR = OUTPUT_DIR / "low_image_weight"

TOP_K = 50

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
# Load data
# ============================================================

print("=" * 70)
print("Extreme Image Weight Analysis")
print("=" * 70)

print("\nLoading gate values...")

gate = np.load(GATE_PATH).reshape(-1)

print("Gate shape:", gate.shape)
print("Gate mean :", gate.mean())
print("Gate min  :", gate.min())
print("Gate max  :", gate.max())

assert len(gate) == 7425


print("\nLoading image mapping...")

with open(IMAGE_MAPPING_PATH, "r") as f:
    iid_to_image = json.load(f)

keep_items = sorted(
    [int(i) for i in iid_to_image.keys()]
)

print("Items with image:", len(keep_items))

assert len(keep_items) == 7425


print("\nLoading item profiles...")

with open(ITEM_PROFILE_PATH, "rb") as f:
    item_profiles = pickle.load(f)

assert len(item_profiles) == 7425


# ============================================================
# Build complete item table
# ============================================================

items = []

for new_id in range(7425):

    old_id = keep_items[new_id]

    gate_value = float(gate[new_id])
    image_weight = 1.0 - gate_value

    profile_data = item_profiles[new_id]

    if isinstance(profile_data, dict):
        profile = profile_data.get("profile", "")
        reasoning = profile_data.get("reasoning", "")
    else:
        profile = str(profile_data)
        reasoning = ""

    items.append({
        "new_id": new_id,
        "old_id": old_id,
        "gate": gate_value,
        "image_weight": image_weight,
        "image_url": iid_to_image[str(old_id)],
        "profile": profile,
        "reasoning": reasoning,
    })


# ============================================================
# Select extremes
# ============================================================

# Largest image weight = smallest text gate
high_image_items = sorted(
    items,
    key=lambda x: x["image_weight"],
    reverse=True
)[:TOP_K]

# Smallest image weight = largest text gate
low_image_items = sorted(
    items,
    key=lambda x: x["image_weight"]
)[:TOP_K]


print("\n" + "=" * 70)
print("TOP 10 LARGEST IMAGE WEIGHTS")
print("=" * 70)

for item in high_image_items[:10]:
    print(
        f"new_id={item['new_id']:4d} | "
        f"old_id={item['old_id']:4d} | "
        f"gate={item['gate']:.6f} | "
        f"image={item['image_weight']:.6f}"
    )


print("\n" + "=" * 70)
print("TOP 10 SMALLEST IMAGE WEIGHTS")
print("=" * 70)

for item in low_image_items[:10]:
    print(
        f"new_id={item['new_id']:4d} | "
        f"old_id={item['old_id']:4d} | "
        f"gate={item['gate']:.6f} | "
        f"image={item['image_weight']:.6f}"
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(items, path):

    fieldnames = [
        "rank",
        "amazon7425_id",
        "original_amazon_id",
        "gate",
        "image_weight",
        "image_url",
        "profile",
        "reasoning",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for rank, item in enumerate(items, 1):

            writer.writerow({
                "rank": rank,
                "amazon7425_id": item["new_id"],
                "original_amazon_id": item["old_id"],
                "gate": item["gate"],
                "image_weight": item["image_weight"],
                "image_url": item["image_url"],
                "profile": item["profile"],
                "reasoning": item["reasoning"],
            })


save_csv(
    high_image_items,
    OUTPUT_DIR / "top50_largest_image_weights.csv"
)

save_csv(
    low_image_items,
    OUTPUT_DIR / "top50_smallest_image_weights.csv"
)


# ============================================================
# Download images
# ============================================================

def download_images(items, output_dir):

    print(f"\nDownloading images to {output_dir}...")

    success = 0
    failed = 0

    for rank, item in enumerate(tqdm(items), 1):

        new_id = item["new_id"]
        old_id = item["old_id"]

        filename = (
            f"{rank:02d}_"
            f"new{new_id}_"
            f"old{old_id}.jpg"
        )

        save_path = output_dir / filename

        item["local_path"] = str(save_path)

        if save_path.exists():
            success += 1
            continue

        try:

            response = requests.get(
                item["image_url"],
                headers=HEADERS,
                timeout=20
            )

            if response.status_code == 200:

                with open(save_path, "wb") as f:
                    f.write(response.content)

                # Verify it can actually be opened
                with Image.open(save_path) as img:
                    img.verify()

                success += 1

            else:

                print(
                    f"\nHTTP {response.status_code}: "
                    f"{item['image_url']}"
                )

                failed += 1

        except Exception as e:

            print(
                f"\nFailed item "
                f"{new_id}: {e}"
            )

            failed += 1

    print("Success:", success)
    print("Failed :", failed)


download_images(
    high_image_items,
    HIGH_IMAGE_DIR
)

download_images(
    low_image_items,
    LOW_IMAGE_DIR
)


# ============================================================
# Contact sheet
# ============================================================

def create_contact_sheet(
    items,
    image_dir,
    output_path,
    title
):

    cols = 5
    rows = 10

    image_width = 180
    image_height = 240
    label_height = 55

    margin = 20
    title_height = 70

    cell_width = image_width + margin
    cell_height = image_height + label_height + margin

    canvas_width = cols * cell_width + margin
    canvas_height = (
        title_height
        + rows * cell_height
        + margin
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf",
            14
        )

        title_font = ImageFont.truetype(
            "DejaVuSans-Bold.ttf",
            24
        )

    except Exception:

        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    draw.text(
        (margin, 20),
        title,
        fill="black",
        font=title_font
    )

    for index, item in enumerate(items):

        row = index // cols
        col = index % cols

        x = margin + col * cell_width
        y = title_height + row * cell_height

        filename = (
            f"{index + 1:02d}_"
            f"new{item['new_id']}_"
            f"old{item['old_id']}.jpg"
        )

        image_path = image_dir / filename

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image.thumbnail(
                (image_width, image_height)
            )

            # Center image inside cell
            paste_x = (
                x
                + (image_width - image.width) // 2
            )

            paste_y = (
                y
                + (image_height - image.height) // 2
            )

            canvas.paste(
                image,
                (paste_x, paste_y)
            )

        except Exception:

            draw.rectangle(
                [
                    x,
                    y,
                    x + image_width,
                    y + image_height
                ],
                outline="black"
            )

            draw.text(
                (x + 10, y + 100),
                "IMAGE FAILED",
                fill="black",
                font=font
            )

        label_y = y + image_height + 5

        label = (
            f"#{index + 1} ID:{item['new_id']}\n"
            f"g={item['gate']:.3f} "
            f"img={item['image_weight']:.3f}"
        )

        draw.text(
            (x, label_y),
            label,
            fill="black",
            font=font
        )

    canvas.save(
        output_path,
        quality=95
    )

    print(
        "Saved contact sheet:",
        output_path
    )


create_contact_sheet(
    high_image_items,
    HIGH_IMAGE_DIR,
    OUTPUT_DIR / "top50_largest_image_weights.jpg",
    "Top 50 Items with Largest Image Weights"
)

create_contact_sheet(
    low_image_items,
    LOW_IMAGE_DIR,
    OUTPUT_DIR / "top50_smallest_image_weights.jpg",
    "Top 50 Items with Smallest Image Weights"
)


print("\n" + "=" * 70)
print("Finished")
print("=" * 70)

print(
    "Results saved to:",
    OUTPUT_DIR
)