"""
SequenceDataset — yields 13-frame windows from cached VAE latents with
per-timestep extras (cur_lin, cur_ang, prev_label_lin, prev_label_ang)
and per-timestep targets (lin, ang). Rejects windows that straddle a
session gap (dt > 1.5 * nominal_dt).
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        seq_len: int = 13,
        nominal_dt: float = 0.5,
        gap_factor: float = 1.5,
    ):
        self.seq_len = seq_len
        self.nominal_dt = nominal_dt
        self.max_dt = nominal_dt * gap_factor

        latents_path = os.path.join(data_root, "latents.npy")
        index_path   = os.path.join(data_root, "index.csv")
        self.latents = np.load(latents_path, mmap_mode="r")

        stamps, lin, ang = [], [], []
        with open(index_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                stamps.append(float(row["stamp"]))
                lin.append(float(row["linear_vel"]))
                ang.append(float(row["angular_vel"]))
        self.stamps = np.array(stamps, dtype=np.float64)
        self.lin = np.array(lin, dtype=np.float32)
        self.ang = np.array(ang, dtype=np.float32)

        assert len(self.stamps) == self.latents.shape[0], \
            "index.csv and latents.npy length mismatch — rebuild the cache"

        N = len(self.stamps)
        dts = np.diff(self.stamps)
        bad = dts > self.max_dt
        valid_starts = []
        for i in range(1, N - seq_len + 1):
            if not bad[i : i + seq_len - 1].any():
                valid_starts.append(i)
        self.valid_starts = np.array(valid_starts, dtype=np.int64)
        print(f"SequenceDataset: {len(self.valid_starts)} valid windows "
              f"({N} total frames, seq_len={seq_len})")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, k: int):
        i = int(self.valid_starts[k])
        T = self.seq_len

        lat = np.array(self.latents[i : i + T], dtype=np.float32)
        tgt = np.stack([self.lin[i : i + T], self.ang[i : i + T]], axis=-1)

        cur_lin = self.lin[i : i + T]
        cur_ang = self.ang[i : i + T]
        prv_lin = self.lin[i - 1 : i + T - 1]
        prv_ang = self.ang[i - 1 : i + T - 1]
        extras = np.stack([cur_lin, cur_ang, prv_lin, prv_ang], axis=-1)

        return (
            torch.from_numpy(lat),
            torch.from_numpy(extras.astype(np.float32)),
            torch.from_numpy(tgt.astype(np.float32)),
        )
