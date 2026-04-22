"""
OrchardNavModel — VAE + GRU + per-timestep MLP (v0.7)
======================================================

    image_i ─► VAE(frozen) ─► (4,32,32) ─► flatten(4096)
                                             │
                                             ▼
                                  Linear(4096→1024) + GELU  ─► z_i (1024)
                                             │
    extras_i (4) ───────────────┐            ▼
                                │       GRU(1 layer, 256)
                                │            │
                                └──► concat [h_i, extras_i]  (260)
                                             │
                                             ▼
                                          MLP head  ─► a_i (2)

Forward accepts EITHER raw images (will encode with the frozen VAE) OR
pre-encoded VAE latents (fast path used during training from disk cache).

Shapes (B = batch, T = seq_len = 13):
    raw images:     (B, T, 3, H, W)    in [-1, 1]
    cached latents: (B, T, 4, 32, 32)
    extras:         (B, T, 4)
    output:         (B, T, 2)
"""

from __future__ import annotations
import torch
import torch.nn as nn


VAE_LATENT_SHAPE = (4, 32, 32)        # sd-vae-ft-mse @ 256x256 input
VAE_LATENT_FLAT = 4 * 32 * 32         # 4096


class OrchardNavModel(nn.Module):
    def __init__(
        self,
        vae_model_id: str = "stabilityai/sd-vae-ft-mse",
        proj_dim: int = 1024,
        hidden_dim: int = 256,
        extra_dim: int = 4,
        action_dim: int = 2,
        seq_len: int = 13,
        load_vae: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.extra_dim = extra_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.vae = None
        if load_vae:
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(vae_model_id)
            for p in self.vae.parameters():
                p.requires_grad_(False)
            self.vae.eval()

        self.proj = nn.Sequential(
            nn.Linear(VAE_LATENT_FLAT, proj_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + extra_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        assert self.vae is not None, "VAE not loaded — pass load_vae=True"
        return self.vae.encode(images).latent_dist.mean

    def forward(
        self,
        latents_or_images: torch.Tensor,
        extras: torch.Tensor,
        is_latents: bool = True,
    ) -> torch.Tensor:
        if is_latents:
            lat = latents_or_images
            B, T = lat.shape[:2]
        else:
            imgs = latents_or_images
            B, T = imgs.shape[:2]
            flat_imgs = imgs.reshape(B * T, *imgs.shape[2:])
            lat = self.encode_images(flat_imgs).reshape(B, T, *VAE_LATENT_SHAPE)

        flat = lat.reshape(B, T, VAE_LATENT_FLAT)
        z = self.proj(flat)                              # (B, T, 1024)
        h_seq, _ = self.gru(z)                           # (B, T, 256)
        fused = torch.cat([h_seq, extras], dim=-1)       # (B, T, 260)
        actions = self.head(fused)                       # (B, T, 2)
        return actions

    def save_trainable(self, path: str):
        state = {
            "proj": self.proj.state_dict(),
            "gru":  self.gru.state_dict(),
            "head": self.head.state_dict(),
            "meta": {
                "proj_dim":   self.proj_dim,
                "hidden_dim": self.hidden_dim,
                "extra_dim":  self.extra_dim,
                "action_dim": self.action_dim,
                "seq_len":    self.seq_len,
            },
        }
        torch.save(state, path)

    def load_trainable(self, path: str, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.proj.load_state_dict(state["proj"])
        self.gru.load_state_dict(state["gru"])
        self.head.load_state_dict(state["head"])
