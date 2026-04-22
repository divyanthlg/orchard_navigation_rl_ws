"""
build_cache — Pre-encode all images with the frozen VAE.

    ros2 run orchard_bc_training build_cache \
        --data_root ~/ros2/orchard_navigation_rl_ws/data/raw
"""

import argparse
import os
import csv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from orchard_bc_training.models import OrchardNavModel, VAE_LATENT_SHAPE
from orchard_bc_training.vae_resolve import resolve_vae_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Folder with labels.csv and images/")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--vae_model_id", default="",
                    help="Leave blank to auto-resolve: use workspace-local "
                         "models/sd-vae-ft-mse if present, else HF id.")
    args = ap.parse_args()

    data_root = os.path.expanduser(args.data_root)
    labels_path = os.path.join(data_root, "labels.csv")
    image_dir   = os.path.join(data_root, "images")
    latents_out = os.path.join(data_root, "latents.npy")
    index_out   = os.path.join(data_root, "index.csv")

    rows = []
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"Found {len(rows)} rows in {labels_path}")

    if "stamp" not in rows[0]:
        raise RuntimeError(
            "labels.csv has no 'stamp' column. Collect with "
            "orchard_bc_training's bc_data_collector, not the v0.6 one.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vae_id = resolve_vae_id(args.vae_model_id)
    print(f"VAE source: {vae_id}")

    model = OrchardNavModel(vae_model_id=vae_id, load_vae=True).to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    N = len(rows)
    latents = np.zeros((N,) + VAE_LATENT_SHAPE, dtype=np.float16)

    batch_imgs, batch_idx = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            z = model.encode_images(x).cpu().numpy().astype(np.float16)
        for k, idx in enumerate(batch_idx):
            latents[idx] = z[k]
        batch_imgs.clear()
        batch_idx.clear()

    for i, row in enumerate(tqdm(rows, desc="Encoding")):
        img_path = os.path.join(image_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        batch_imgs.append(tfm(img))
        batch_idx.append(i)
        if len(batch_imgs) >= args.batch_size:
            flush()
    flush()

    np.save(latents_out, latents)
    print(f"Saved latents: {latents_out}  shape={latents.shape}")

    has_odom_stamp = "odom_stamp" in rows[0]
    if not has_odom_stamp:
        print("NOTE: labels.csv has no odom_stamp column (pre-v0.7.3). "
              "Writing index.csv with odom_stamp = stamp for back-compat.")

    with open(index_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "filename", "stamp", "odom_stamp",
                    "linear_vel", "angular_vel"])
        for i, r in enumerate(rows):
            odom_stamp = r["odom_stamp"] if has_odom_stamp else r["stamp"]
            w.writerow([i, r["filename"], r["stamp"], odom_stamp,
                        r["linear_vel"], r["angular_vel"]])
    print(f"Saved index:   {index_out}")


if __name__ == "__main__":
    main()
