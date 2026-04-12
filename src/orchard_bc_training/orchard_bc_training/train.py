"""
train — BC training on cached VAE latents.

    ros2 run orchard_bc_training train \
        --data_root ~/ros2/orchard_navigation_rl_ws/data/raw \
        --out_dir   ~/ros2/orchard_navigation_rl_ws/checkpoints \
        --epochs 50 --batch_size 64 --lr 3e-4
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from orchard_bc_training.models import OrchardNavModel
from orchard_bc_training.datasets import SequenceDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq_len", type=int, default=13)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    data_root = os.path.expanduser(args.data_root)
    out_dir   = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = SequenceDataset(data_root, seq_len=args.seq_len)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_trn = len(ds) - n_val
    trn, val = random_split(ds, [n_trn, n_val],
                             generator=torch.Generator().manual_seed(0))
    trn_loader = DataLoader(trn, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = OrchardNavModel(load_vae=False, seq_len=args.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        trn_loss = 0.0
        for lat, extras, tgt in trn_loader:
            lat = lat.to(device, non_blocking=True)
            extras = extras.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            pred = model(lat, extras, is_latents=True)
            loss = loss_fn(pred, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trn_loss += loss.item() * lat.size(0)
        trn_loss /= len(trn_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lat, extras, tgt in val_loader:
                lat = lat.to(device, non_blocking=True)
                extras = extras.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                pred = model(lat, extras, is_latents=True)
                val_loss += loss_fn(pred, tgt).item() * lat.size(0)
        val_loss /= len(val_loader.dataset)

        sched.step()
        print(f"Epoch {epoch:3d} | train {trn_loss:.5f} | val {val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            model.save_trainable(os.path.join(out_dir, "best.pt"))
            print(f"  ↳ new best, saved best.pt")

    model.save_trainable(os.path.join(out_dir, "last.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
