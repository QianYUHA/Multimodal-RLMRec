import os
import sys
import numpy as np
import torch

# Make sure imports such as config.xxx work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


def main():

    print("=" * 60)
    print("Gate Analysis - LightGCN_plus_gated_openai_clip_image")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Build data handler
    # --------------------------------------------------
    print("\n[1] Loading Amazon7425 dataset...")

    data_handler = build_data_handler()
    data_handler.load_data()

    print("Users :", configs['data']['user_num'])
    print("Items :", configs['data']['item_num'])

    # --------------------------------------------------
    # 2. Build model
    # --------------------------------------------------
    print("\n[2] Building model...")

    model = build_model(data_handler)
    model = model.to(configs['device'])

    # --------------------------------------------------
    # 3. Load checkpoint
    # --------------------------------------------------
    ckpt_path = (
        "./encoder/checkpoint/"
        "lightgcn_plus_gated_openai_clip_image/"
        "lightgcn_plus_gated_openai_clip_image-amazon7425-2023.pth"
    )

    print("\n[3] Loading checkpoint:")
    print(ckpt_path)

    checkpoint = torch.load(
        ckpt_path,
        map_location=configs['device']
    )

    print("Checkpoint type:", type(checkpoint))

    # Some projects save state_dict directly,
    # while others save a dictionary containing state_dict.
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    print("PASS: Checkpoint loaded successfully.")

    # --------------------------------------------------
    # 4. Evaluation mode
    # --------------------------------------------------
    model.eval()

    # --------------------------------------------------
    # 5. Extract semantic embeddings
    # --------------------------------------------------
    print("\n[4] Computing projected semantic embeddings...")

    with torch.no_grad():

        text_embeds = model.text_projection(
            model.item_text_embeds
        )

        image_embeds = model.image_projection(
            model.item_image_embeds
        )

        fused_embeds, gate_values = model.fusion(
            text_embeds,
            image_embeds
        )

    print("Text projected shape :", text_embeds.shape)
    print("Image projected shape:", image_embeds.shape)
    print("Fused shape          :", fused_embeds.shape)
    print("Gate shape           :", gate_values.shape)

    # --------------------------------------------------
    # 6. Convert gate to numpy
    # --------------------------------------------------
    gates = gate_values.squeeze(-1).cpu().numpy()

    print("\n[5] Gate statistics")
    print("-" * 40)

    print("Number of items :", len(gates))
    print("Mean            :", gates.mean())
    print("Std             :", gates.std())
    print("Min             :", gates.min())
    print("25%             :", np.percentile(gates, 25))
    print("Median          :", np.median(gates))
    print("75%             :", np.percentile(gates, 75))
    print("Max             :", gates.max())

    # --------------------------------------------------
    # 7. Text / image dominant items
    # --------------------------------------------------
    text_dominant = gates > 0.5
    image_dominant = gates < 0.5
    balanced = gates == 0.5

    print("\n[6] Modality preference")
    print("-" * 40)

    print(
        "Text-dominant items :",
        text_dominant.sum(),
        f"({text_dominant.mean() * 100:.2f}%)"
    )

    print(
        "Image-dominant items:",
        image_dominant.sum(),
        f"({image_dominant.mean() * 100:.2f}%)"
    )

    print(
        "Exactly balanced    :",
        balanced.sum(),
        f"({balanced.mean() * 100:.2f}%)"
    )

    # --------------------------------------------------
    # 8. Save gates
    # --------------------------------------------------
    output_dir = "./generation/analysis"
    os.makedirs(output_dir, exist_ok=True)

    output_path = (
        f"{output_dir}/"
        "gated_openai_clip_image_amazon7425_gate.npy"
    )

    np.save(output_path, gates)

    print("\n[7] Saved gate values:")
    print(output_path)

    print("\n" + "=" * 60)
    print("Gate analysis completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()