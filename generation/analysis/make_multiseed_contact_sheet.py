from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re


BASE = Path(
    "generation/analysis/"
    "extreme_image_weights_multiseed"
)

GROUPS = {
    "high_image_weight":
        "Top 50 Highest Mean Image Weights",
    "low_image_weight":
        "Top 50 Lowest Mean Image Weights",
}

COLS = 10
ROWS = 5

CELL_W = 180
CELL_H = 290

IMAGE_W = 150
IMAGE_H = 210

TOP_MARGIN = 50


def parse_filename(name):
    pattern = (
        r"(\d+)_new(\d+)_old(\d+)_"
        r"w([\d.]+)_std([\d.]+)"
    )

    match = re.search(pattern, name)

    if not match:
        return None

    return {
        "rank": int(match.group(1)),
        "new_id": int(match.group(2)),
        "old_id": int(match.group(3)),
        "weight": float(match.group(4)),
        "std": float(match.group(5)),
    }


def make_sheet(folder_name, title):

    folder = BASE / folder_name

    files = sorted(
        folder.glob("*.jpg")
    )

    assert len(files) == 50, (
        f"Expected 50 images, got {len(files)}"
    )

    sheet_w = COLS * CELL_W
    sheet_h = TOP_MARGIN + ROWS * CELL_H

    sheet = Image.new(
        "RGB",
        (sheet_w, sheet_h),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf",
            14
        )

        small_font = ImageFont.truetype(
            "DejaVuSans.ttf",
            11
        )

        title_font = ImageFont.truetype(
            "DejaVuSans-Bold.ttf",
            22
        )

    except:
        font = ImageFont.load_default()
        small_font = font
        title_font = font

    draw.text(
        (20, 12),
        title,
        fill="black",
        font=title_font
    )

    for idx, path in enumerate(files):

        info = parse_filename(
            path.stem
        )

        row = idx // COLS
        col = idx % COLS

        x = col * CELL_W
        y = TOP_MARGIN + row * CELL_H

        with Image.open(path) as img:

            img = img.convert("RGB")

            img.thumbnail(
                (IMAGE_W, IMAGE_H)
            )

            paste_x = (
                x +
                (CELL_W - img.width) // 2
            )

            paste_y = y + 5

            sheet.paste(
                img,
                (paste_x, paste_y)
            )

        if info:

            text_y = y + IMAGE_H + 12

            draw.text(
                (x + 8, text_y),
                (
                    f"#{info['rank']} "
                    f"id={info['new_id']}"
                ),
                fill="black",
                font=font
            )

            draw.text(
                (x + 8, text_y + 19),
                (
                    f"w={info['weight']:.4f} "
                    f"std={info['std']:.4f}"
                ),
                fill="black",
                font=small_font
            )

    output = BASE / (
        f"{folder_name}_contact_sheet.jpg"
    )

    sheet.save(
        output,
        quality=95
    )

    print("Saved:", output)


for folder_name, title in GROUPS.items():

    make_sheet(
        folder_name,
        title
    )