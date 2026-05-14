"""
Resolve the VAE model ID: prefer a workspace-local snapshot, fall back to the
HuggingFace identifier. This makes the pipeline work fully offline if the user
has copied the VAE into `<workspace>/models/sd-vae-ft-mse/`.
"""

import os


DEFAULT_WORKSPACE_VAE = os.path.expanduser(
    '~/ros2/orchard_navigation_rl_ws/models/sd-vae-ft-mse')

DEFAULT_HF_ID = 'stabilityai/sd-vae-ft-mse'


def resolve_vae_id(user_value: str = '') -> str:
    """
    If user_value is non-empty, return it unchanged (user knows best).
    Otherwise prefer the workspace-local path if it exists, else the HF id.
    """
    if user_value:
        return os.path.expanduser(user_value)
    if os.path.isdir(DEFAULT_WORKSPACE_VAE):
        return DEFAULT_WORKSPACE_VAE
    return DEFAULT_HF_ID
