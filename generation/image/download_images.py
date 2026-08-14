import json
import requests
from pathlib import Path
from tqdm import tqdm

SAVE_DIR = Path("data/book_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

mapping = json.load(open("iid_to_image.json"))

headers = {
    "User-Agent": "Mozilla/5.0"
}

MAX_IMAGES = 100

success = 0
failed = 0

for iid, url in tqdm(list(mapping.items())[:MAX_IMAGES]):

    save_path = SAVE_DIR / f"{iid}.jpg"

    if save_path.exists():
        continue

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        if response.status_code == 200:

            with open(save_path, "wb") as f:
                f.write(response.content)

            success += 1

        else:

            failed += 1

    except Exception:

        failed += 1

print("--------------------------------")
print("Success:", success)
print("Failed :", failed)