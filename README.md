# Orchard Navigation RL Workspace

ROS 2 workspace for orchard navigation behavior cloning on a Clearpath Warthog.

## Packages
- `orchard_data_collector` — v0.6 data collection (single-frame BC baseline).
- `orchard_nav_deploy` — v0.6 policy deploy + DAgger.
- `orchard_bc_training` — v0.7 VAE+GRU sequence BC (collect, train, deploy, viz).

## Setup
```bash
cd ~/ros2/orchard_navigation_rl_ws
colcon build --symlink-install
source install/setup.bash
```

Python deps (system Python, needed by ROS):
```bash
/usr/bin/python3 -m pip install --user 'Pillow>=10' diffusers tqdm torchvision
```

Pre-download the VAE once with internet (saves to `models/`):
```bash
/usr/bin/python3 -c "
from diffusers import AutoencoderKL
m = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
m.save_pretrained('models/sd-vae-ft-mse')
"
```

See `orchard_bc_training/CHEATSHEET.md` for the full v0.7 workflow.

## Not in this repo (regenerate locally)
- `build/`, `install/`, `log/` — `colcon build`.
- `data/` — record with `bc_collect.launch.py` or `collect.launch.py`.
- `checkpoints/` — produced by `ros2 run orchard_bc_training train`.
- `models/` — downloaded from HuggingFace as above.
