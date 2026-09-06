import pickle
import numpy as np

DATA_DIR = "data/amazon7425"

trn = pickle.load(open(f"{DATA_DIR}/trn_mat.pkl", "rb"))
val = pickle.load(open(f"{DATA_DIR}/val_mat.pkl", "rb"))
tst = pickle.load(open(f"{DATA_DIR}/tst_mat.pkl", "rb"))

usr_prf = pickle.load(open(f"{DATA_DIR}/usr_prf.pkl", "rb"))
itm_prf = pickle.load(open(f"{DATA_DIR}/itm_prf.pkl", "rb"))

usr_emb = pickle.load(open(f"{DATA_DIR}/usr_emb_np.pkl", "rb"))
itm_emb = pickle.load(open(f"{DATA_DIR}/itm_emb_np.pkl", "rb"))

print("=" * 60)
print("Amazon7425 Dataset Sanity Check")
print("=" * 60)

print("\nInteraction matrices:")
print("Train:", trn.shape, trn.nnz)
print("Val  :", val.shape, val.nnz)
print("Test :", tst.shape, tst.nnz)

print("\nProfiles:")
print("Users:", len(usr_prf))
print("Items:", len(itm_prf))

print("\nEmbeddings:")
print("User:", usr_emb.shape)
print("Item:", itm_emb.shape)

# Shape checks
assert trn.shape == (10989, 7425)
assert val.shape == (10989, 7425)
assert tst.shape == (10989, 7425)

assert len(usr_prf) == 10989
assert len(itm_prf) == 7425

assert usr_emb.shape == (10989, 1536)
assert itm_emb.shape == (7425, 1536)

print("\nPASS: Dataset shapes are correct.")