import json
import gzip

# 1. 读取 iid -> asin
asin_to_iid = {}

with open("data/mapper/amazon_item.json", "r") as f:
    for line in f:
        obj = json.loads(line)

        iid = obj["iid"]
        asin = obj["asin"]

        asin_to_iid[asin] = iid

print("Mapper items:", len(asin_to_iid))

# 2. 匹配 metadata
iid_to_image = {}

matched = 0

with gzip.open("meta_Books.jsonl.gz.1", "rt", encoding="utf-8") as f:
    for line in f:

        item = json.loads(line)

        asin = item["parent_asin"]

        if asin not in asin_to_iid:
            continue

        matched += 1

        images = item.get("images", [])

        if len(images) == 0:
            continue

        image_url = images[0]["large"]

        iid = asin_to_iid[asin]

        iid_to_image[iid] = image_url

print("Matched:", matched)
print("With image:", len(iid_to_image))

# 3. 保存
with open("iid_to_image.json", "w") as f:
    json.dump(iid_to_image, f)

print("Saved iid_to_image.json")