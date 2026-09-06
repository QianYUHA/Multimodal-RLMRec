from config.configurator import configs
from data_utils.build_data_handler import build_data_handler

print("=" * 60)
print("DataHandler Sanity Check")
print("=" * 60)

print("Dataset:", configs["data"]["name"])

data_handler = build_data_handler()
data_handler.load_data()

print("\nInteraction matrices:")
print("Train:", data_handler.trn_mat.shape)
print("Train interactions:", data_handler.trn_mat.nnz)

print("\nConfig:")
print("Users:", configs["data"]["user_num"])
print("Items:", configs["data"]["item_num"])

print("\nTorch adjacency:")
print("Shape:", data_handler.torch_adj.shape)

assert data_handler.trn_mat.shape == (10989, 7425)

assert configs["data"]["user_num"] == 10989
assert configs["data"]["item_num"] == 7425

assert data_handler.torch_adj.shape == (
    10989 + 7425,
    10989 + 7425
)

print("\nPASS: DataHandler works correctly.")