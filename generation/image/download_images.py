import json
import time
import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


# ============================================================
# Config
# ============================================================

MAPPING_PATH = Path("iid_to_image.json")
SAVE_DIR = Path("data/book_images")

FAILED_LOG = Path(
    "generation/image/failed_image_downloads.json"
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)

# Connect timeout, read timeout
TIMEOUT = (10, 30)

# Small delay to avoid hitting the server too aggressively
REQUEST_DELAY = 0.05


# ============================================================
# HTTP Session with Retry
# ============================================================

session = requests.Session()

retry_strategy = Retry(
    total=4,
    connect=4,
    read=4,
    status=4,
    backoff_factor=1.0,
    status_forcelist=[
        429,
        500,
        502,
        503,
        504,
    ],
    allowed_methods=["GET"],
    raise_on_status=False,
)

adapter = HTTPAdapter(
    max_retries=retry_strategy
)

session.mount("http://", adapter)
session.mount("https://", adapter)

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": (
        "image/avif,image/webp,image/apng,"
        "image/svg+xml,image/*,*/*;q=0.8"
    ),
})


# ============================================================
# Helper Functions
# ============================================================
def get_candidate_urls(url: str):
    """
    Generate fallback URLs for Amazon image links.

    Example:
    xxx._SX322_BO1,204,203,200_.jpg
    ->
    xxx.jpg
    """

    candidates = [url]

    # Remove Amazon image transformation suffix:
    # ._SX322_BO1,204,203,200_.jpg -> .jpg
    base_url = re.sub(
        r"\._[^/]*_\.jpg$",
        ".jpg",
        url,
        flags=re.IGNORECASE,
    )

    if base_url != url:
        candidates.append(base_url)

    # Optional common Amazon resized variants
    if base_url.lower().endswith(".jpg"):
        stem = base_url[:-4]

        candidates.append(
            stem + "._SL1000_.jpg"
        )

        candidates.append(
            stem + "._SL500_.jpg"
        )

    # Remove duplicates while preserving order
    return list(dict.fromkeys(candidates))

def is_valid_image(path: Path) -> bool:
    """
    Check whether an existing image file is readable.
    """
    if not path.exists():
        return False

    try:
        with Image.open(path) as img:
            img.verify()
        return True

    except Exception:
        return False


def download_image(iid: str, url: str, save_path: Path):
    """
    Download one image.

    If the original Amazon transformed URL fails,
    automatically try the base/original image URL
    and common Amazon size variants.

    Returns the URL that successfully downloaded.
    """

    errors = []

    for candidate_url in get_candidate_urls(url):

        try:

            response = session.get(
                candidate_url,
                timeout=TIMEOUT,
            )

            if response.status_code != 200:
                errors.append(
                    f"{candidate_url} -> "
                    f"HTTP {response.status_code}"
                )
                continue

            if not response.content:
                errors.append(
                    f"{candidate_url} -> "
                    "empty response"
                )
                continue

            # Validate downloaded bytes
            try:
                image = Image.open(
                    BytesIO(response.content)
                )
                image.load()
                image = image.convert("RGB")

            except Exception as e:
                errors.append(
                    f"{candidate_url} -> "
                    f"invalid image: {e}"
                )
                continue

            temp_path = save_path.with_suffix(
                ".tmp.jpg"
            )

            image.save(
                temp_path,
                format="JPEG",
                quality=95,
            )

            if not is_valid_image(temp_path):

                if temp_path.exists():
                    temp_path.unlink()

                errors.append(
                    f"{candidate_url} -> "
                    "saved image failed validation"
                )
                continue

            temp_path.replace(save_path)

            return candidate_url

        except Exception as e:

            errors.append(
                f"{candidate_url} -> {e}"
            )

    raise RuntimeError(
        "All candidate URLs failed: "
        + " | ".join(errors)
    )


# ============================================================
# Load Mapping
# ============================================================

print("=" * 70)
print("Download Amazon Book Images")
print("=" * 70)

with open(
    MAPPING_PATH,
    "r",
    encoding="utf-8",
) as f:
    mapping = json.load(f)

print("Total mapped items:", len(mapping))

if len(mapping) != 7425:
    print(
        "WARNING: expected 7425 mapped items, "
        f"but found {len(mapping)}"
    )


# ============================================================
# Download
# ============================================================

downloaded = 0
skipped = 0
repaired = 0
failed = []

for iid, url in tqdm(
    mapping.items(),
    total=len(mapping),
    desc="Downloading",
):

    save_path = SAVE_DIR / f"{iid}.jpg"

    # --------------------------------------------------------
    # Resume support:
    # skip existing valid images
    # --------------------------------------------------------

    if save_path.exists():

        if is_valid_image(save_path):
            skipped += 1
            continue

        else:
            # Existing file is corrupt
            print(
                f"\nCorrupt existing image: "
                f"{save_path}"
            )

            save_path.unlink()
            repaired += 1

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    try:

        working_url = download_image(
        iid=iid,
        url=url,
        save_path=save_path,
        )

        downloaded += 1

        # If fallback URL worked, update mapping
        if working_url != url:

            print(
                f"\nFallback succeeded for iid={iid}"
            )

            print(
                f"Old: {url}"
            )

            print(
                f"New: {working_url}"
            )

            mapping[iid] = working_url

    except Exception as e:

        failed.append({
            "iid": iid,
            "url": url,
            "error": str(e),
        })

        print(
            f"\nFAILED iid={iid}: {e}"
        )

    time.sleep(REQUEST_DELAY)

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
# Save Failure Log
# ============================================================

with open(
    FAILED_LOG,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        failed,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Final Validation
# ============================================================

print("\n" + "=" * 70)
print("Validating downloaded dataset")
print("=" * 70)

valid = 0
invalid = []

for iid in tqdm(
    mapping.keys(),
    desc="Validating",
):

    path = SAVE_DIR / f"{iid}.jpg"

    if is_valid_image(path):
        valid += 1

    else:
        invalid.append(iid)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("Download Summary")
print("=" * 70)

print(
    f"Mapped items       : {len(mapping)}"
)
print(
    f"Already valid      : {skipped}"
)
print(
    f"Newly downloaded   : {downloaded}"
)
print(
    f"Corrupt replaced   : {repaired}"
)
print(
    f"Download failures  : {len(failed)}"
)

print("-" * 70)

print(
    f"Valid images now   : "
    f"{valid}/{len(mapping)}"
)

print(
    f"Missing / invalid  : "
    f"{len(invalid)}"
)

print(
    f"Failure log        : "
    f"{FAILED_LOG}"
)

if invalid:
    print(
        "\nFirst invalid IDs:"
    )
    print(
        invalid[:20]
    )

if valid == len(mapping):

    print("\nSUCCESS:")
    print(
        "All mapped item images are "
        "downloaded and valid."
    )

else:

    print("\nWARNING:")
    print(
        "Some images are still missing. "
        "Run this script again to retry them."
    )

print("=" * 70)