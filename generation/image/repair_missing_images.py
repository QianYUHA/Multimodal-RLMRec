import json
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


# ============================================================
# Config
# ============================================================

SAVE_DIR = Path("data/book_images")
MAPPING_PATH = Path("iid_to_image.json")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

TIMEOUT = 30


# ============================================================
# Four permanently broken Amazon URLs
#
# old_id -> metadata
# ============================================================

MISSING = {
    "7497": {
        "title": "Fooled by Randomness",
        "isbn": "9781400067930",
    },
    "4687": {
        "title": "A Widow for One Year",
        "isbn": "9780375501371",
    },
    "8519": {
        "title": "The Plantagenets",
        "isbn": "9780143124924",
    },
    "4842": {
        "title": "Just One Look",
        "isbn": "9780451235039",
    },
}


# ============================================================
# Helpers
# ============================================================

def save_image_from_url(url, save_path):

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    image = Image.open(
        BytesIO(response.content)
    )

    image.load()
    image = image.convert("RGB")

    temp_path = save_path.with_suffix(".tmp.jpg")

    image.save(
        temp_path,
        format="JPEG",
        quality=95,
    )

    # verify
    with Image.open(temp_path) as img:
        img.verify()

    temp_path.replace(save_path)


def google_books_cover(isbn):

    api_url = (
        "https://www.googleapis.com/books/v1/volumes"
        f"?q=isbn:{isbn}"
    )

    response = requests.get(
        api_url,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("totalItems", 0) == 0:
        return None

    items = data.get("items", [])

    for item in items:

        info = item.get(
            "volumeInfo",
            {}
        )

        links = info.get(
            "imageLinks",
            {}
        )

        # Prefer higher resolution
        for key in [
            "extraLarge",
            "large",
            "medium",
            "small",
            "thumbnail",
            "smallThumbnail",
        ]:

            url = links.get(key)

            if url:

                # Google sometimes returns http
                url = url.replace(
                    "http://",
                    "https://"
                )

                return url

    return None


def openlibrary_cover(isbn):

    return (
        "https://covers.openlibrary.org/"
        f"b/isbn/{isbn}-L.jpg"
    )


# ============================================================
# Load original mapping
# ============================================================

with open(
    MAPPING_PATH,
    "r",
    encoding="utf-8",
) as f:

    mapping = json.load(f)


repair_log = {}


# ============================================================
# Repair
# ============================================================

for iid, metadata in MISSING.items():

    title = metadata["title"]
    isbn = metadata["isbn"]

    save_path = (
        SAVE_DIR / f"{iid}.jpg"
    )

    print("\n" + "=" * 70)
    print(f"IID   : {iid}")
    print(f"Title : {title}")
    print(f"ISBN  : {isbn}")
    print("=" * 70)

    success = False

    # --------------------------------------------------------
    # 1. Google Books
    # --------------------------------------------------------

    try:

        url = google_books_cover(
            isbn
        )

        if url:

            print(
                "Trying Google Books:"
            )
            print(url)

            save_image_from_url(
                url,
                save_path,
            )

            print(
                "SUCCESS via Google Books"
            )

            repair_log[iid] = {
                "title": title,
                "isbn": isbn,
                "source": "Google Books",
                "replacement_url": url,
            }

            mapping[iid] = url

            success = True

    except Exception as e:

        print(
            "Google Books failed:",
            e
        )

    # --------------------------------------------------------
    # 2. Open Library fallback
    # --------------------------------------------------------

    if not success:

        try:

            url = openlibrary_cover(
                isbn
            )

            print(
                "Trying Open Library:"
            )
            print(url)

            save_image_from_url(
                url,
                save_path,
            )

            print(
                "SUCCESS via Open Library"
            )

            repair_log[iid] = {
                "title": title,
                "isbn": isbn,
                "source": "Open Library",
                "replacement_url": url,
            }

            mapping[iid] = url

            success = True

        except Exception as e:

            print(
                "Open Library failed:",
                e
            )

    if not success:

        print(
            "FAILED TO REPAIR"
        )


# ============================================================
# Save updated mapping
# ============================================================

with open(
    MAPPING_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        mapping,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Save repair documentation
# ============================================================

LOG_PATH = Path(
    "generation/image/"
    "replacement_image_log.json"
)

with open(
    LOG_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        repair_log,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Verify
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)

for iid, metadata in MISSING.items():

    path = (
        SAVE_DIR / f"{iid}.jpg"
    )

    try:

        with Image.open(path) as img:

            print(
                iid,
                metadata["title"],
                "-> OK",
                img.size
            )

    except Exception as e:

        print(
            iid,
            metadata["title"],
            "-> FAILED",
            e
        )

print("\nRepair log:")
print(LOG_PATH)