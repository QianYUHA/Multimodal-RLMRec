import os
import sys
import csv

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

# Make sure imports such as config.xxx work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


MODEL_NAME = "lightgcn_plus_gated_openai_clip_image"
DATASET_NAME = "amazon7425"
SEEDS = [2023, 2024, 2025]

CHECKPOINT_DIR = (
    "./encoder/checkpoint/"
    "lightgcn_plus_gated_openai_clip_image"
)

OUTPUT_DIR = "./generation/analysis"


def load_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    else:
        return checkpoint


def extract_gates(model, seed):
    ckpt_path = os.path.join(
        CHECKPOINT_DIR,
        f"{MODEL_NAME}-{DATASET_NAME}-{seed}.pth"
    )

    print(f"\nLoading seed {seed}:")
    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}"
        )

    checkpoint = torch.load(
        ckpt_path,
        map_location=configs["device"]
    )

    state_dict = load_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        text_embeds = model.text_projection(
            model.item_text_embeds
        )

        image_embeds = model.image_projection(
            model.item_image_embeds
        )

        _, gate_values = model.fusion(
            text_embeds,
            image_embeds
        )

    gates = (
        gate_values
        .squeeze(-1)
        .detach()
        .cpu()
        .numpy()
    )

    if not np.isfinite(gates).all():
        raise ValueError(
            f"Non-finite gate values found for seed {seed}"
        )

    print(
        f"Seed {seed}: "
        f"mean={gates.mean():.6f}, "
        f"std={gates.std():.6f}, "
        f"min={gates.min():.6f}, "
        f"max={gates.max():.6f}"
    )

    return gates


def print_distribution(name, values):
    print(f"\n{name}")
    print("-" * 50)
    print("N      :", len(values))
    print("Mean   :", values.mean())
    print("Std    :", values.std())
    print("Min    :", values.min())
    print("25%    :", np.percentile(values, 25))
    print("Median :", np.median(values))
    print("75%    :", np.percentile(values, 75))
    print("Max    :", values.max())


def main():

    print("=" * 70)
    print("Multi-seed Gate Analysis")
    print("OpenAI Text + CLIP Image + Gated Fusion")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------

    print("\n[1] Loading dataset...")

    data_handler = build_data_handler()
    data_handler.load_data()

    print("Users:", configs["data"]["user_num"])
    print("Items:", configs["data"]["item_num"])

    # --------------------------------------------------
    # 2. Model
    # --------------------------------------------------

    print("\n[2] Building model...")

    model = build_model(data_handler)
    model = model.to(configs["device"])

    # --------------------------------------------------
    # 3. Extract gate values from every seed
    # --------------------------------------------------

    print("\n[3] Extracting gate values...")

    gates_by_seed = []

    for seed in SEEDS:
        gates = extract_gates(model, seed)
        gates_by_seed.append(gates)

    gates_by_seed = np.stack(
        gates_by_seed,
        axis=0
    )

    print(
        "\nCombined gate array shape:",
        gates_by_seed.shape
    )

    if gates_by_seed.shape[1] != configs["data"]["item_num"]:
        raise ValueError(
            "Gate item count does not match dataset item count."
        )

    # --------------------------------------------------
    # 4. Mean and standard deviation across seeds
    # --------------------------------------------------

    gate_mean = gates_by_seed.mean(axis=0)

    # sample standard deviation across the 3 seeds
    gate_std = gates_by_seed.std(
        axis=0,
        ddof=1
    )

    image_weight_by_seed = 1.0 - gates_by_seed
    image_weight_mean = 1.0 - gate_mean

    image_weight_std = image_weight_by_seed.std(
        axis=0,
        ddof=1
    )

    print_distribution(
        "[4] Mean text-gate distribution",
        gate_mean
    )

    print_distribution(
        "[5] Mean image-weight distribution",
        image_weight_mean
    )

    # --------------------------------------------------
    # 5. Modality preference
    # --------------------------------------------------

    text_dominant = gate_mean > 0.5
    image_dominant = gate_mean < 0.5
    balanced = np.isclose(gate_mean, 0.5)

    print("\n[6] Mean modality preference")
    print("-" * 50)

    print(
        "Text-dominant :",
        text_dominant.sum(),
        f"({text_dominant.mean() * 100:.2f}%)"
    )

    print(
        "Image-dominant:",
        image_dominant.sum(),
        f"({image_dominant.mean() * 100:.2f}%)"
    )

    print(
        "Balanced      :",
        balanced.sum(),
        f"({balanced.mean() * 100:.2f}%)"
    )

    # --------------------------------------------------
    # 6. Cross-seed gate consistency
    # --------------------------------------------------

    print("\n[7] Cross-seed gate consistency")
    print("-" * 50)

    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):

            a = gates_by_seed[i]
            b = gates_by_seed[j]

            pearson_r, pearson_p = pearsonr(a, b)
            spearman_r, spearman_p = spearmanr(a, b)

            print(
                f"{SEEDS[i]} vs {SEEDS[j]} | "
                f"Pearson={pearson_r:.6f} "
                f"(p={pearson_p:.3e}) | "
                f"Spearman={spearman_r:.6f} "
                f"(p={spearman_p:.3e})"
            )

    # --------------------------------------------------
    # 7. Save arrays
    # --------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prefix = os.path.join(
        OUTPUT_DIR,
        "gated_openai_clip_image_amazon7425_multiseed"
    )

    np.save(
        prefix + "_gates_by_seed.npy",
        gates_by_seed
    )

    np.save(
        prefix + "_gate_mean.npy",
        gate_mean
    )

    np.save(
        prefix + "_gate_std.npy",
        gate_std
    )

    # --------------------------------------------------
    # 8. Save item-level CSV
    # --------------------------------------------------

    csv_path = prefix + "_items.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "item_id",
            "gate_2023",
            "gate_2024",
            "gate_2025",
            "gate_mean",
            "gate_std",
            "image_weight_2023",
            "image_weight_2024",
            "image_weight_2025",
            "image_weight_mean",
            "image_weight_std",
        ])

        for item_id in range(len(gate_mean)):

            writer.writerow([
                item_id,
                gates_by_seed[0, item_id],
                gates_by_seed[1, item_id],
                gates_by_seed[2, item_id],
                gate_mean[item_id],
                gate_std[item_id],
                image_weight_by_seed[0, item_id],
                image_weight_by_seed[1, item_id],
                image_weight_by_seed[2, item_id],
                image_weight_mean[item_id],
                image_weight_std[item_id],
            ])

    # --------------------------------------------------
    # 9. Save top / bottom 50 image-weight items
    # --------------------------------------------------

    sorted_indices = np.argsort(image_weight_mean)

    bottom50 = sorted_indices[:50]
    top50 = sorted_indices[-50:][::-1]

    def save_subset(path, indices):

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "rank",
                "item_id",
                "gate_mean",
                "gate_std",
                "image_weight_mean",
                "image_weight_std",
            ])

            for rank, item_id in enumerate(
                indices,
                start=1
            ):

                writer.writerow([
                    rank,
                    item_id,
                    gate_mean[item_id],
                    gate_std[item_id],
                    image_weight_mean[item_id],
                    image_weight_std[item_id],
                ])

    save_subset(
        prefix + "_top50_image_weight.csv",
        top50
    )

    save_subset(
        prefix + "_bottom50_image_weight.csv",
        bottom50
    )

    print("\n[8] Saved outputs")
    print("-" * 50)
    print(prefix + "_gates_by_seed.npy")
    print(prefix + "_gate_mean.npy")
    print(prefix + "_gate_std.npy")
    print(csv_path)
    print(prefix + "_top50_image_weight.csv")
    print(prefix + "_bottom50_image_weight.csv")

    print("\nGate analysis completed.")


if __name__ == "__main__":
    main()