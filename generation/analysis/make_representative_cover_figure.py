from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(
    "generation/analysis/"
    "extreme_image_weights_multiseed"
)

HIGH_DIR = (
    BASE_DIR /
    "high_image_weight"
)

LOW_DIR = (
    BASE_DIR /
    "low_image_weight"
)

OUTPUT_DIR = Path(
    "generation/analysis/"
    "thesis_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

N_EXAMPLES = 6

CELL_W = 220
IMAGE_H = 280
LABEL_H = 55

TITLE_H = 60

ROWS = 2
COLS = N_EXAMPLES

SHEET_W = (
    COLS *
    CELL_W
)

SHEET_H = (
    TITLE_H +
    ROWS *
    (IMAGE_H + LABEL_H)
)


# ============================================================
# Filename parser
# ============================================================

pattern = re.compile(
    r"(\d+)_"
    r"new(\d+)_"
    r"old(\d+)_"
    r"w([\d.]+)_"
    r"std([\d.]+)"
)


def load_items(folder):

    items = []

    for path in folder.glob("*.jpg"):

        match = pattern.search(
            path.stem
        )

        if match is None:
            continue

        items.append({
            "path":
                path,

            "rank":
                int(match.group(1)),

            "new_id":
                int(match.group(2)),

            "old_id":
                int(match.group(3)),

            "weight":
                float(match.group(4)),

            "std":
                float(match.group(5)),
        })

    return items


# ============================================================
# Select stable representative examples
# ============================================================

high_items = load_items(
    HIGH_DIR
)

low_items = load_items(
    LOW_DIR
)

assert len(high_items) == 50
assert len(low_items) == 50


# Transparent selection rule:
# lowest cross-seed standard deviation
# within each extreme group.

high_selected = sorted(
    high_items,
    key=lambda x: x["std"]
)[:N_EXAMPLES]

low_selected = sorted(
    low_items,
    key=lambda x: x["std"]
)[:N_EXAMPLES]


print("Selected HIGH-image-weight examples:")

for x in high_selected:
    print(
        x["new_id"],
        x["weight"],
        x["std"]
    )


print("\nSelected LOW-image-weight examples:")

for x in low_selected:
    print(
        x["new_id"],
        x["weight"],
        x["std"]
    )


# ============================================================
# Fonts
# ============================================================

try:

    title_font = ImageFont.truetype(
        "DejaVuSans-Bold.ttf",
        24
    )

    row_font = ImageFont.truetype(
        "DejaVuSans-Bold.ttf",
        17
    )

    label_font = ImageFont.truetype(
        "DejaVuSans.ttf",
        13
    )

except Exception:

    title_font = ImageFont.load_default()
    row_font = title_font
    label_font = title_font


# ============================================================
# Create canvas
# ============================================================

sheet = Image.new(
    "RGB",
    (
        SHEET_W,
        SHEET_H
    ),
    "white"
)

draw = ImageDraw.Draw(
    sheet
)


draw.text(
    (20, 15),
    "Representative Books with High and Low Learned Image Weights",
    fill="black",
    font=title_font
)


def paste_row(
    items,
    row,
    row_title
):

    y_start = (
        TITLE_H +
        row *
        (IMAGE_H + LABEL_H)
    )

    draw.text(
        (
            10,
            y_start + 5
        ),
        row_title,
        fill="black",
        font=row_font
    )

    image_y = (
        y_start + 30
    )

    for col, item in enumerate(
        items
    ):

        x = (
            col *
            CELL_W
        )

        with Image.open(
            item["path"]
        ) as img:

            img = img.convert(
                "RGB"
            )

            img.thumbnail(
                (
                    CELL_W - 30,
                    IMAGE_H - 35
                )
            )

            paste_x = (
                x +
                (
                    CELL_W -
                    img.width
                ) // 2
            )

            paste_y = (
                image_y +
                (
                    IMAGE_H -
                    35 -
                    img.height
                ) // 2
            )

            sheet.paste(
                img,
                (
                    paste_x,
                    paste_y
                )
            )

        label_y = (
            y_start +
            IMAGE_H
        )

        text = (
            f"ID {item['new_id']}\n"
            f"w={item['weight']:.3f}, "
            f"SD={item['std']:.3f}"
        )

        draw.multiline_text(
            (
                x + 12,
                label_y
            ),
            text,
            fill="black",
            font=label_font,
            spacing=2
        )


paste_row(
    high_selected,
    0,
    "Higher image weight"
)

paste_row(
    low_selected,
    1,
    "Lower image weight"
)


# ============================================================
# Save
# ============================================================

output_path = (
    OUTPUT_DIR /
    "fig_representative_high_low_image_weight_covers.png"
)

sheet.save(
    output_path,
    dpi=(300, 300)
)

print("\nSaved:", output_path)