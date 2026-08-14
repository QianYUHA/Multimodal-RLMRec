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