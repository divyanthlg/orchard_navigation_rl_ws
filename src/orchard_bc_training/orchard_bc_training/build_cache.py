"""
build_cache — Pre-encode all images with the frozen VAE.

Discovers every session_* subfolder under <data_root>/sessions/, merges
their labels.csv files in chronological order (by folder name, which is
timestamp-based), then encodes all images in a single pass.

    ros2 run orchard_bc_training build_cache \
        --data_root ~/ros2/orchard_navigation_rl_ws/data/raw

Outputs (unchanged contract with train / datasets.py):
    <data_root>/latents.npy
    <data_root>/index.csv
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


def _discover_sessions(data_root: str) -> list:
    """Return sorted list of session_* folder paths under data_root/sessions/."""
    sessions_root = os.path.join(data_root, 'sessions')
    if not os.path.isdir(sessions_root):
        raise RuntimeError(
            f'No sessions/ directory found under {data_root}.\n'
            f'Expected: {sessions_root}\n'
            f'Have you collected any data yet?')
    sessions = sorted([
        os.path.join(sessions_root, d)
        for d in os.listdir(sessions_root)
        if d.startswith('session_')
        and os.path.isdir(os.path.join(sessions_root, d))
    ])
    if not sessions:
        raise RuntimeError(
            f'No session_* folders found in {sessions_root}.\n'
            f'Have you collected any data yet?')
    return sessions


def _load_session_rows(session_dir: str) -> list:
    """Load and validate rows from one session's labels.csv."""
    labels_path = os.path.join(session_dir, 'labels.csv')
    image_dir   = os.path.join(session_dir, 'images')

    if not os.path.isfile(labels_path):
        print(f'  WARNING: no labels.csv in {session_dir} — skipping')
        return []

    rows = []
    with open(labels_path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print(f'  WARNING: labels.csv in {session_dir} is empty — skipping')
        return []

    if 'stamp' not in rows[0]:
        print(f'  WARNING: labels.csv in {session_dir} has no "stamp" column — skipping')
        return []

    # Attach absolute image paths so the caller does not need session_dir again
    for r in rows:
        r['_abs_image_path'] = os.path.join(image_dir, r['filename'])

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True,
                    help='Parent folder that contains the sessions/ subdirectory')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--vae_model_id', default='',
                    help='Leave blank to auto-resolve: use workspace-local '
                         'models/sd-vae-ft-mse if present, else HF id.')
    args = ap.parse_args()

    data_root   = os.path.expanduser(args.data_root)
    latents_out = os.path.join(data_root, 'latents.npy')
    index_out   = os.path.join(data_root, 'index.csv')

    # ── 1. Discover and merge sessions ────────────────────────────────
    sessions = _discover_sessions(data_root)
    print(f'Found {len(sessions)} session(s):')
    all_rows = []
    for s in sessions:
        rows = _load_session_rows(s)
        print(f'  {os.path.basename(s)}: {len(rows)} frames')
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError('No valid rows across all sessions. Nothing to encode.')

    print(f'\nTotal frames to encode: {len(all_rows)}')

    # ── 2. Load VAE + model ───────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    vae_id = resolve_vae_id(args.vae_model_id)
    print(f'VAE source: {vae_id}')

    model = OrchardNavModel(vae_model_id=vae_id, load_vae=True).to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # ── 3. Encode in batches ──────────────────────────────────────────
    N = len(all_rows)
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

    for i, row in enumerate(tqdm(all_rows, desc='Encoding')):
        img = Image.open(row['_abs_image_path']).convert('RGB')
        batch_imgs.append(tfm(img))
        batch_idx.append(i)
        if len(batch_imgs) >= args.batch_size:
            flush()
    flush()

    np.save(latents_out, latents)
    print(f'Saved latents: {latents_out}  shape={latents.shape}')

    # ── 4. Write merged index.csv ─────────────────────────────────────
    with open(index_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'filename', 'stamp', 'odom_stamp',
                    'linear_vel', 'angular_vel'])
        for i, r in enumerate(all_rows):
            has_odom   = 'odom_stamp' in r
            odom_stamp = r['odom_stamp'] if has_odom else r['stamp']
            w.writerow([i, r['filename'], r['stamp'], odom_stamp,
                        r['linear_vel'], r['angular_vel']])

    print(f'Saved index:   {index_out}')
    print(f'\nDone — {N} frames from {len(sessions)} session(s) encoded.')


if __name__ == '__main__':
    main()
