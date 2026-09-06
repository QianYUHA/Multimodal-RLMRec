import json
import pickle
import numpy as np
import scipy.sparse as sp

# ============================
# Load interaction matrices
# ============================


trn = pickle.load(open("data/amazon/trn_mat.pkl", "rb")).tocsr()
val = pickle.load(open("data/amazon/val_mat.pkl", "rb")).tocsr()
tst = pickle.load(open("data/amazon/tst_mat.pkl", "rb")).tocsr()

print(type(trn))
print(type(val))
print(type(tst))

print("=" * 50)
print("Original Dataset")
print("=" * 50)

print("Users :", trn.shape[0])
print("Items :", trn.shape[1])

print("Train interactions :", trn.nnz)
print("Validation interactions :", val.nnz)
print("Test interactions :", tst.nnz)

# ============================
# Load image mapping
# ============================

with open("iid_to_image.json", "r") as f:
    iid_to_image = json.load(f)

keep_items = sorted([int(i) for i in iid_to_image.keys()])

print("\nItems with image :", len(keep_items))
print("First 10 item ids:")
print(keep_items[:10])

print("\nUnique item ids:", len(set(keep_items)))
print("Min item id:", min(keep_items))
print("Max item id:", max(keep_items))

# ============================
# Check missing items
# ============================

missing = set(range(9332)) - set(keep_items)

print("\nMissing item number:", len(missing))
print("First 20 missing ids:")
print(sorted(list(missing))[:20])

# ============================
# Find users to keep
# ============================

# First keep only image items
trn_keep = trn[:, keep_items]
val_keep = val[:, keep_items]
tst_keep = tst[:, keep_items]

# Count interactions for each user
user_degree = (
    trn_keep.getnnz(axis=1)
    + val_keep.getnnz(axis=1)
    + tst_keep.getnnz(axis=1)
)

keep_users = np.where(user_degree > 0)[0]

print("\n==============================")
print("Users after filtering")
print("==============================")

print("Original users:", trn.shape[0])
print("Remaining users:", len(keep_users))
print("Removed users:", trn.shape[0] - len(keep_users))

# ============================
# Build ID mapping
# ============================

print("\n==============================")
print("Build New ID Mapping")
print("==============================")

user_mapping = {
    old: new
    for new, old in enumerate(keep_users)
}

item_mapping = {
    old: new
    for new, old in enumerate(keep_items)
}

print("New users:", len(user_mapping))
print("New items:", len(item_mapping))

print()

print("First 5 user mappings:")

for i, (k, v) in enumerate(user_mapping.items()):
    print(k, "->", v)
    if i == 4:
        break

print()

print("First 5 item mappings:")

for i, (k, v) in enumerate(item_mapping.items()):
    print(k, "->", v)
    if i == 4:
        break

def remap_matrix(mat, keep_users, keep_items, user_mapping, item_mapping):
    """
    Filter and re-index a sparse interaction matrix.
    """

    # Keep only selected users and items
    filtered = mat[keep_users][:, keep_items]

    # Convert to COO for easy row/column remapping
    coo = filtered.tocoo()

    # Original local indices -> new indices
    new_rows = coo.row
    new_cols = coo.col

    # Since filtered matrix is already ordered according to
    # keep_users / keep_items, these are already new indices.

    new_mat = sp.coo_matrix(
        (coo.data, (new_rows, new_cols)),
        shape=(len(keep_users), len(keep_items))
    ).tocsr()

    return new_mat

new_trn = remap_matrix(
    trn,
    keep_users,
    keep_items,
    user_mapping,
    item_mapping
)

new_val = remap_matrix(
    val,
    keep_users,
    keep_items,
    user_mapping,
    item_mapping
)

new_tst = remap_matrix(
    tst,
    keep_users,
    keep_items,
    user_mapping,
    item_mapping
)

print("\n==============================")
print("New Interaction Matrices")
print("==============================")

print("Train:")
print("Shape:", new_trn.shape)
print("Interactions:", new_trn.nnz)

print("\nValidation:")
print("Shape:", new_val.shape)
print("Interactions:", new_val.nnz)

print("\nTest:")
print("Shape:", new_tst.shape)
print("Interactions:", new_tst.nnz)

assert new_trn.shape == new_val.shape
assert new_trn.shape == new_tst.shape

assert new_trn.shape == (
    len(keep_users),
    len(keep_items)
)

print("\nPASS: All matrices have consistent shape.")

assert new_trn.nnz <= trn.nnz
assert new_val.nnz <= val.nnz
assert new_tst.nnz <= tst.nnz

print("PASS: Filtering did not create new interactions.")

# ============================
# Load semantic profiles
# ============================

usr_prf = pickle.load(
    open("data/amazon/usr_prf.pkl", "rb")
)

itm_prf = pickle.load(
    open("data/amazon/itm_prf.pkl", "rb")
)

usr_emb = pickle.load(
    open("data/amazon/usr_emb_np.pkl", "rb")
)

itm_emb = pickle.load(
    open("data/amazon/itm_emb_np.pkl", "rb")
)

print("\n==============================")
print("Original Semantic Data")
print("==============================")

print("usr_prf:", type(usr_prf), len(usr_prf))
print("itm_prf:", type(itm_prf), len(itm_prf))

print("usr_emb:", type(usr_emb), usr_emb.shape)
print("itm_emb:", type(itm_emb), itm_emb.shape)

# ============================
# Filter item semantic data
# ============================

new_itm_prf = {
    new_id: itm_prf[old_id]
    for new_id, old_id in enumerate(keep_items)
}

new_itm_emb = itm_emb[keep_items]

print("\n==============================")
print("New Item Semantic Data")
print("==============================")

print("new_itm_prf:", len(new_itm_prf))
print("new_itm_emb:", new_itm_emb.shape)

# ============================
# Filter user semantic data
# ============================

new_usr_prf = {
    new_id: usr_prf[old_id]
    for new_id, old_id in enumerate(keep_users)
}

new_usr_emb = usr_emb[keep_users]

print("\n==============================")
print("New User Semantic Data")
print("==============================")

print("new_usr_prf:", len(new_usr_prf))
print("new_usr_emb:", new_usr_emb.shape)

# ============================
# Verify ID alignment
# ============================

for new_id in range(5):

    old_id = keep_items[new_id]

    assert new_itm_prf[new_id] == itm_prf[old_id]
    assert np.allclose(
        new_itm_emb[new_id],
        itm_emb[old_id]
    )

print("\nPASS: Item semantic data aligned correctly.")

for new_id in range(5):

    old_id = keep_users[new_id]

    assert new_usr_prf[new_id] == usr_prf[old_id]
    assert np.allclose(
        new_usr_emb[new_id],
        usr_emb[old_id]
    )

print("PASS: User semantic data aligned correctly.")

# ============================
# Save amazon7425 dataset
# ============================

import os

OUTPUT_DIR = "data/amazon7425"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n==============================")
print("Saving amazon7425")
print("==============================")

# Interaction matrices
with open(f"{OUTPUT_DIR}/trn_mat.pkl", "wb") as f:
    pickle.dump(new_trn, f)

with open(f"{OUTPUT_DIR}/val_mat.pkl", "wb") as f:
    pickle.dump(new_val, f)

with open(f"{OUTPUT_DIR}/tst_mat.pkl", "wb") as f:
    pickle.dump(new_tst, f)

# User semantic data
with open(f"{OUTPUT_DIR}/usr_prf.pkl", "wb") as f:
    pickle.dump(new_usr_prf, f)

with open(f"{OUTPUT_DIR}/usr_emb_np.pkl", "wb") as f:
    pickle.dump(new_usr_emb, f)

# Item semantic data
with open(f"{OUTPUT_DIR}/itm_prf.pkl", "wb") as f:
    pickle.dump(new_itm_prf, f)

with open(f"{OUTPUT_DIR}/itm_emb_np.pkl", "wb") as f:
    pickle.dump(new_itm_emb, f)

print("Saved to:", OUTPUT_DIR)