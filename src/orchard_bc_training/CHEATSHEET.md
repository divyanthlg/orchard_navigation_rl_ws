# orchard_bc_training — Cheat Sheet (v0.7.1)

Workspace: `~/ros2/orchard_navigation_rl_ws`

## Pipeline

```
  [1] bc_collect.launch.py ──► data/raw/images/ + labels.csv
  [2] build_cache          ──► data/raw/latents.npy + index.csv
  [3] train                ──► checkpoints/best.pt
  [4] bc_deploy.launch.py  ──► robot drives itself + /bc_viz/image overlay
```

---

## One-time setup

### Install package
```bash
cd ~/ros2/orchard_navigation_rl_ws/src
rm -rf orchard_bc_training
unzip ~/Downloads/orchard_bc_training.zip
cd ~/ros2/orchard_navigation_rl_ws
colcon build --symlink-install --packages-select orchard_bc_training
source install/setup.bash
```

### Python deps (system Python, NOT pyenv)
```bash
/usr/bin/python3 -m pip install --user 'Pillow>=10' diffusers tqdm torchvision
```

### Pre-download the VAE (needs internet — do this once, off the robot WiFi)
```bash
# Option A: HuggingFace cache (simpler)
/usr/bin/python3 -c "
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
print('cached at ~/.cache/huggingface/')
"

# Option B: Workspace-local copy (more robust — survives cache clears)
mkdir -p ~/ros2/orchard_navigation_rl_ws/models
SNAP=$(ls ~/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/ | head -1)
cp -rL ~/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/$SNAP \
       ~/ros2/orchard_navigation_rl_ws/models/sd-vae-ft-mse
```

After Option B, the package auto-detects `models/sd-vae-ft-mse/` and uses it
with zero config. No internet ever needed again.

### Every new terminal
```bash
source ~/ros2/orchard_navigation_rl_ws/install/setup.bash
```

The launch files themselves set `HF_HUB_OFFLINE=1` automatically, so Warthog
WiFi (no internet) is fine.

---

## Step 1 — Collect

```bash
# Terminal A
ros2 launch orchard_bc_training bc_collect.launch.py

# Terminal B
ros2 service call /bc_data_collector/toggle_recording \
    std_srvs/srv/SetBool "{data: true}"
# drive the robot...
ros2 service call /bc_data_collector/toggle_recording \
    std_srvs/srv/SetBool "{data: false}"
```

Key args: `save_rate_hz` (default 2.0, don't change), `auto_start`,
`image_dir`, `labels_file`.

Multiple sessions append to the same CSV. Session gaps are detected at
dataset-build time via the `stamp` column, so no training window crosses a
gap.

---

## Step 2 — Build cache

```bash
ros2 run orchard_bc_training build_cache \
    --data_root ~/ros2/orchard_navigation_rl_ws/data/raw
```

Passes every image through the frozen VAE once, saves `latents.npy` (float16)
+ `index.csv`. Re-run whenever `labels.csv` grows.

Args: `--data_root` (required), `--batch_size` (32), `--vae_model_id`
(empty → auto-resolve local or HF cache).

---

## Step 3 — Train

```bash
ros2 run orchard_bc_training train \
    --data_root ~/ros2/orchard_navigation_rl_ws/data/raw \
    --out_dir   ~/ros2/orchard_navigation_rl_ws/checkpoints \
    --epochs 50 --batch_size 64 --lr 3e-4
```

Reads cached latents (VAE not loaded → fast, low VRAM). Trains
Linear(4096→1024) → GRU(256) → MLP head with per-timestep MSE on 13-frame
windows. Saves `best.pt` + `last.pt`.

Args: `--epochs 50`, `--batch_size 64` (→16 for dummy data),
`--lr 3e-4`, `--seq_len 13` (must match deploy).

---

## Step 4 — Deploy + visualize

```bash
# Terminal A: launches policy + mux + status display + viz node
ros2 launch orchard_bc_training bc_deploy.launch.py

# Terminal B: view the annotated camera stream
ros2 run rqt_image_view rqt_image_view /bc_viz/image
```

**Nodes launched:**
| Node | Purpose |
|---|---|
| `bc_policy_node` | Loads `best.pt`, VAE encode @ 2 Hz, publishes `/bc_policy/cmd_vel` @ 10 Hz |
| `bc_cmd_vel_mux` | Forwards policy → `/w200_0100/cmd_vel`, RC overrides instantly |
| `bc_status_display` | Terminal indicator: `[POLICY]` / `[HUMAN]` + numeric cmd |
| `bc_viz_node` | Overlays command bars on camera, publishes `/bc_viz/image` |

**What the overlay shows:**

| Element | Meaning |
|---|---|
| **Top bar (ANG)** | Angular velocity. Blue marker = policy output. White marker = current odom. Full-scale ±0.5 rad/s. |
| **Bottom bar (LIN)** | Linear velocity. Same markers. Full-scale ±1.0 m/s. |
| **Top-left tag** | Green `[POLICY]` or red `[HUMAN]` depending on what the mux is forwarding to the robot. |

When the blue and white markers overlap, the robot is tracking the policy's
intent. When blue drifts left/right of white, the policy is commanding a
change (turn) the robot hasn't yet executed — this is expected during
steering. When the tag is red, you've grabbed the RC.

**First ~6.5 s after launch** publishes zeros while the 13-frame buffer
fills. Expected.

**Override paths at launch time:**
```bash
ros2 launch orchard_bc_training bc_deploy.launch.py \
    checkpoint_path:=$HOME/somewhere/last.pt \
    vae_model_id:=$HOME/other_vae_snapshot
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Hangs at `Loading model on cuda...` | No internet + no local VAE. Pre-download (Option A) or copy snapshot to `models/sd-vae-ft-mse/` (Option B). |
| `PIL.Image has no attribute 'Resampling'` | `/usr/bin/python3 -m pip install --user 'Pillow>=10'` |
| `No such file: best.pt` | Override `checkpoint_path:=...` or move checkpoint. |
| `No such file: latents.npy` | Re-run step 2 with matching `--data_root`. |
| `0 valid windows` during train | All sessions shorter than 6.5 s. Record longer continuous runs. |
| Viz image blank / not updating | `ros2 topic hz /bc_viz/image` — 0 Hz means camera stream isn't arriving. |
| Policy keeps publishing zeros | Buffer not filling. `ros2 topic hz <image_topic>`. |
| Two different Pythons (pyenv vs system) | Always use `/usr/bin/python3 -m pip install --user ...` for ROS deps. |

---

## Topics / services reference

**Published:** `/bc_policy/cmd_vel`, `/bc_policy/active`, `/w200_0100/cmd_vel`,
`/bc_mux/active_source`, `/bc_viz/image`.

**Subscribed:** `/camera/camera/color/image_raw`,
`/w200_0100/platform/odom/filtered`, `/w200_0100/rc_teleop/cmd_vel`.

**Services:** `/bc_data_collector/toggle_recording`,
`/bc_data_collector/status`.
