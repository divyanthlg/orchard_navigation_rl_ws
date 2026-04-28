# orchard_bc_training — Complete Terminal Cheatsheet (v0.7.4)

> **Workspace root:** `~/ros2/orchard_navigation_rl_ws`
> **Default topics (v0.7.4):**
> | Signal | Topic |
> |---|---|
> | Camera (compressed) | `/sensors/camera_0/color/compressed` |
> | Odometry | `/platform/odom/filtered` |
> | RC / human override | `/rc_teleop/cmd_vel` |
> | Robot cmd output | `/cmd_vel` |

---

## Table of Contents

1. [One-Time Setup](#1-one-time-setup)
2. [Every New Terminal](#2-every-new-terminal)
3. [Step 1 — Collect Training Data](#3-step-1--collect-training-data)
4. [Step 2 — Build Latent Cache](#4-step-2--build-latent-cache)
5. [Step 3 — Train the Policy](#5-step-3--train-the-policy)
6. [Step 4 — Deploy the Policy](#6-step-4--deploy-the-policy)
7. [Topic Sanity Checks](#7-topic-sanity-checks)
8. [Override Reference (all launch args)](#8-override-reference-all-launch-args)
9. [Sync Quality Check](#9-sync-quality-check)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. One-Time Setup

### 1.1 Install the package

```bash
cd ~/ros2/orchard_navigation_rl_ws/src

# Remove any old copy
rm -rf orchard_bc_training

# Unzip the new v0.7.4 package
unzip ~/Downloads/orchard_bc_training.zip

# Build (symlink-install lets you edit Python files without rebuilding)
cd ~/ros2/orchard_navigation_rl_ws
colcon build --symlink-install --packages-select orchard_bc_training

# Source the workspace
source install/setup.bash
```

### 1.2 Install Python dependencies

> Always use the **system Python** (`/usr/bin/python3`), NOT pyenv.
> ROS 2 nodes use the system interpreter; mixing them causes import errors.

```bash
/usr/bin/python3 -m pip install --user 'Pillow>=10' diffusers tqdm torchvision
```

| Package | Why |
|---|---|
| `Pillow>=10` | Image resizing in `build_cache` (older versions lack `Resampling`) |
| `diffusers` | Loads the Stable Diffusion VAE (`AutoencoderKL`) |
| `tqdm` | Progress bar during cache building |
| `torchvision` | Image transforms for the policy node |

### 1.3 Pre-download the VAE (do once, needs internet)

The VAE (`stabilityai/sd-vae-ft-mse`) is ~335 MB. Download it before going
to the field where the robot has no internet.

**Option A — HuggingFace cache (simpler)**

```bash
/usr/bin/python3 -c "
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
print('Cached at ~/.cache/huggingface/')
"
```

**Option B — Workspace-local copy (recommended — survives cache clears)**

```bash
# First run Option A above to populate the HF cache, then:
mkdir -p ~/ros2/orchard_navigation_rl_ws/models

SNAP=$(ls ~/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/ | head -1)

cp -rL \
  ~/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/$SNAP \
  ~/ros2/orchard_navigation_rl_ws/models/sd-vae-ft-mse
```

After Option B, the package **auto-detects** `models/sd-vae-ft-mse/` at
startup — no config needed. Both launch files set `HF_HUB_OFFLINE=1`
automatically so robot WiFi (no internet) is never an issue.

---

## 2. Every New Terminal

```bash
source ~/ros2/orchard_navigation_rl_ws/install/setup.bash
```

> Put this in `~/.bashrc` to run it automatically:
> ```bash
> echo "source ~/ros2/orchard_navigation_rl_ws/install/setup.bash" >> ~/.bashrc
> ```

---

## 3. Step 1 — Collect Training Data

### 3.1 Launch the collector

```bash
# Terminal A
ros2 launch orchard_bc_training bc_collect.launch.py
```

This starts `bc_data_collector`, which:
- Subscribes to `/sensors/camera_0/color/compressed` and `/platform/odom/filtered`
- Synchronises image + odom messages with a 100 ms slop window
- Saves 256×256 PNG images to `data/raw/images/`
- Appends rows to `data/raw/labels.csv` at 2 Hz

### 3.2 Start and stop recording

```bash
# Terminal B — start recording
ros2 service call /bc_data_collector/toggle_recording \
    std_srvs/srv/SetBool "{data: true}"

# Drive the robot around the orchard...

# Stop recording
ros2 service call /bc_data_collector/toggle_recording \
    std_srvs/srv/SetBool "{data: false}"
```

### 3.3 Check recording status

```bash
ros2 service call /bc_data_collector/status std_srvs/srv/Trigger "{}"
```

Returns: `State: RECORDING | Session: <N> | Total: <M>`

### 3.4 Multiple sessions

Running the collector again and toggling recording will **append** new
frames to the existing `labels.csv`. The gap between sessions is
automatically detected at training time via the `stamp` column — no
data from two separate drives will be joined into a single training window.

### Launch arguments for `bc_collect.launch.py`

| Argument | Default | Description |
|---|---|---|
| `image_topic` | `/sensors/camera_0/color/compressed` | CompressedImage topic to subscribe to |
| `odom_topic` | `/platform/odom/filtered` | Odometry topic |
| `save_rate_hz` | `2.0` | Max frame save rate — **do not increase** (matches training) |
| `auto_start` | `false` | Set `true` to begin recording immediately on launch |
| `skip_stationary` | `false` | Set `true` to discard frames where `|linear_vel| < min_linear_vel` |
| `image_dir` | `<ws>/data/raw/images` | Where PNG frames are saved |
| `labels_file` | `<ws>/data/raw/labels.csv` | CSV path |

**Example — auto-start with a custom topic:**

```bash
ros2 launch orchard_bc_training bc_collect.launch.py \
    auto_start:=true \
    image_topic:=/sensors/camera_0/color/compressed
```

---

## 4. Step 2 — Build Latent Cache

Run this **once after every collection session** (or whenever `labels.csv` grows).
It passes every saved PNG through the frozen VAE and writes compressed float16
latent vectors to disk, so training never needs to load the VAE.

```bash
ros2 run orchard_bc_training build_cache \
    --data_root ~/ros2/orchard_navigation_rl_ws/data/raw
```

**What it produces:**

| File | Description |
|---|---|
| `data/raw/latents.npy` | Float16 array, shape `(N, 4, 32, 32)` — one latent per frame |
| `data/raw/index.csv` | Indexed copy of `labels.csv` with `idx, filename, stamp, odom_stamp, linear_vel, angular_vel` |

### Arguments for `build_cache`

| Argument | Default | Description |
|---|---|---|
| `--data_root` | *(required)* | Folder containing `labels.csv` and `images/` |
| `--image_size` | `256` | Resize images to this square size before VAE encoding |
| `--batch_size` | `32` | Number of images per GPU batch — reduce if you run out of VRAM |
| `--vae_model_id` | *(auto)* | Leave blank: auto-resolves to workspace-local VAE or HF cache |

**Example — smaller batch for low-VRAM machine:**

```bash
ros2 run orchard_bc_training build_cache \
    --data_root ~/ros2/orchard_navigation_rl_ws/data/raw \
    --batch_size 8
```

> Re-run `build_cache` every time you collect more data. It re-encodes **all**
> frames, so `latents.npy` is always consistent with the current `labels.csv`.

---

## 5. Step 3 — Train the Policy

Reads `latents.npy` and `index.csv` (VAE not loaded — fast, low VRAM).
Trains `Linear(4096→1024) → GRU(256) → MLP(2)` with per-timestep MSE loss
on 13-frame sliding windows.

```bash
ros2 run orchard_bc_training train \
    --data_root ~/ros2/orchard_navigation_rl_ws/data/raw \
    --out_dir   ~/ros2/orchard_navigation_rl_ws/checkpoints \
    --epochs 50 \
    --batch_size 64 \
    --lr 3e-4
```

**What it produces:**

| File | Description |
|---|---|
| `checkpoints/best.pt` | Checkpoint with lowest validation loss — use this for deploy |
| `checkpoints/last.pt` | Checkpoint from the final epoch |

### Arguments for `train`

| Argument | Default | Description |
|---|---|---|
| `--data_root` | *(required)* | Folder with `latents.npy` and `index.csv` |
| `--out_dir` | *(required)* | Where to save `best.pt` / `last.pt` |
| `--epochs` | `50` | Number of full training passes |
| `--batch_size` | `64` | Sequences per gradient step — reduce to `16` on CPU/low VRAM |
| `--lr` | `3e-4` | AdamW learning rate |
| `--seq_len` | `13` | Frames per training window — **must match deploy** |
| `--val_frac` | `0.1` | Fraction of windows held out for validation |
| `--num_workers` | `4` | DataLoader worker processes — set `0` to disable multiprocessing |

**Example — quick test run on CPU:**

```bash
ros2 run orchard_bc_training train \
    --data_root ~/ros2/orchard_navigation_rl_ws/data/raw \
    --out_dir   ~/ros2/orchard_navigation_rl_ws/checkpoints \
    --epochs 10 \
    --batch_size 16 \
    --num_workers 0
```

> **Tip:** Watch the `val` loss column. If it stops improving before epoch 50,
> training is done. If it's still falling at epoch 50, increase `--epochs`.

---

## 6. Step 4 — Deploy the Policy

### 6.1 Launch the full deploy stack

```bash
# Terminal A — starts all four nodes
ros2 launch orchard_bc_training bc_deploy.launch.py
```

**Nodes launched:**

| Node | Role |
|---|---|
| `bc_policy_node` | Decodes compressed images → VAE encode @ 2 Hz → fills 13-frame ring buffer → GRU+MLP inference → publishes `/bc_policy/cmd_vel` @ 10 Hz |
| `bc_cmd_vel_mux` | Merges policy and RC commands; RC instantly overrides, releases back to policy after 0.5 s of silence |
| `bc_status_display` | Prints live `[POLICY]` / `[HUMAN]` indicator + numeric velocities to terminal |
| `bc_viz_node` | Overlays velocity bars on the camera stream, publishes `/bc_viz/image` |

### 6.2 View the annotated camera overlay

```bash
# Terminal B
ros2 run rqt_image_view rqt_image_view /bc_viz/image
```

**Overlay guide:**

| Element | Meaning |
|---|---|
| **Top bar (ANG)** | Angular velocity. Blue marker = policy output. White marker = current odom. Full-scale ±0.5 rad/s |
| **Bottom bar (LIN)** | Linear velocity. Blue = policy, white = odom. Full-scale ±1.0 m/s |
| **Top-left tag GREEN** | `[POLICY]` — robot following the learned policy |
| **Top-left tag RED** | `[HUMAN]` — RC override active |

> **First ~6.5 s after launch** the policy publishes zeros while the 13-frame
> buffer fills. This is expected — do not drive during this warmup.

### Launch arguments for `bc_deploy.launch.py`

| Argument | Default | Description |
|---|---|---|
| `image_topic` | `/sensors/camera_0/color/compressed` | CompressedImage input |
| `odom_topic` | `/platform/odom/filtered` | Odometry input |
| `human_cmd_topic` | `/rc_teleop/cmd_vel` | RC teleop override topic |
| `output_cmd_topic` | `/cmd_vel` | Final velocity command sent to the robot |
| `checkpoint_path` | `<ws>/checkpoints/best.pt` | Policy weights to load |
| `vae_model_id` | *(auto)* | Leave blank to auto-resolve workspace-local or HF cache |

**Example — load a specific checkpoint:**

```bash
ros2 launch orchard_bc_training bc_deploy.launch.py \
    checkpoint_path:=$HOME/ros2/orchard_navigation_rl_ws/checkpoints/last.pt
```

**Example — override all topics:**

```bash
ros2 launch orchard_bc_training bc_deploy.launch.py \
    image_topic:=/sensors/camera_0/color/compressed \
    odom_topic:=/platform/odom/filtered \
    human_cmd_topic:=/rc_teleop/cmd_vel \
    output_cmd_topic:=/cmd_vel \
    checkpoint_path:=$HOME/somewhere/best.pt
```

---

## 7. Topic Sanity Checks

Run these before collecting or deploying to confirm topics are live.

```bash
# Confirm the camera is publishing CompressedImage (not raw Image)
ros2 topic info /sensors/camera_0/color/compressed
# Expected type: sensor_msgs/msg/CompressedImage

# Check camera frame rate (should be ≥ 2 Hz)
ros2 topic hz /sensors/camera_0/color/compressed

# Check odometry is arriving
ros2 topic hz /platform/odom/filtered

# Confirm the RC teleop topic exists
ros2 topic hz /rc_teleop/cmd_vel

# Monitor the policy output during deploy
ros2 topic echo /bc_policy/cmd_vel

# Monitor what is actually sent to the robot (policy or human)
ros2 topic echo /cmd_vel

# Check which source the mux is currently forwarding
ros2 topic echo /bc_mux/active_source
```

---

## 8. Override Reference (all launch args)

### `bc_collect.launch.py` — full override example

```bash
ros2 launch orchard_bc_training bc_collect.launch.py \
    image_topic:=/sensors/camera_0/color/compressed \
    odom_topic:=/platform/odom/filtered \
    save_rate_hz:=2.0 \
    auto_start:=false \
    skip_stationary:=false \
    image_dir:=$HOME/ros2/orchard_navigation_rl_ws/data/raw/images \
    labels_file:=$HOME/ros2/orchard_navigation_rl_ws/data/raw/labels.csv
```

### `bc_deploy.launch.py` — full override example

```bash
ros2 launch orchard_bc_training bc_deploy.launch.py \
    image_topic:=/sensors/camera_0/color/compressed \
    odom_topic:=/platform/odom/filtered \
    human_cmd_topic:=/rc_teleop/cmd_vel \
    output_cmd_topic:=/cmd_vel \
    checkpoint_path:=$HOME/ros2/orchard_navigation_rl_ws/checkpoints/best.pt \
    vae_model_id:=$HOME/ros2/orchard_navigation_rl_ws/models/sd-vae-ft-mse
```

---

## 9. Sync Quality Check

Run this after a collection session to verify image and odom timestamps are
well-aligned. A large `stamp - odom_stamp` difference indicates a sync
problem that will degrade training quality.

```bash
cd ~/ros2/orchard_navigation_rl_ws/data/raw

python3 - <<'EOF'
import pandas as pd
df = pd.read_csv('labels.csv')
if 'odom_stamp' in df.columns:
    diff_ms = (df['stamp'] - df['odom_stamp']) * 1000
    print(diff_ms.describe())
    print(f"\nMax absolute diff: {diff_ms.abs().max():.1f} ms")
    print("Target: |mean| and std well under 100 ms")
    if diff_ms.abs().quantile(0.99) > 100:
        print("WARNING: p99 > 100 ms — consider loosening sync_slop_sec or checking topic QoS")
    else:
        print("OK: sync quality looks good")
else:
    print("No odom_stamp column found (pre-v0.7.3 file)")
EOF
```

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Hangs at `Loading model on cuda...` | No internet and no local VAE copy | Run Option B in §1.3 to copy VAE into `models/sd-vae-ft-mse/` |
| `cv2.imdecode returned None` on every frame | Topic publishes non-JPEG/PNG data or zero-byte payloads | Run `ros2 topic echo /sensors/camera_0/color/compressed --no-arr` and inspect the `.format` field |
| `No frames arriving` / 0 Hz on image topic | Topic name mismatch or camera driver not publishing compressed variant | `ros2 topic list \| grep camera` to find the actual topic name |
| `PIL.Image has no attribute 'Resampling'` | Pillow version too old | `/usr/bin/python3 -m pip install --user 'Pillow>=10'` |
| `No such file: best.pt` | Checkpoint not found at default path | Use `checkpoint_path:=<absolute/path/to/best.pt>` at deploy launch |
| `No such file: latents.npy` | Cache not built or wrong `--data_root` | Re-run Step 2 with the correct `--data_root` |
| `0 valid windows` during train | All recorded sessions shorter than 6.5 s (< 13 frames at 2 Hz) | Record longer continuous runs; each session needs > 13 frames |
| `/bc_viz/image` blank / not updating | Compressed stream not arriving or decode failing | `ros2 topic hz /sensors/camera_0/color/compressed` — if 0 Hz, fix the camera driver |
| Policy keeps publishing zeros after warmup | Ring buffer not filling — image topic silent | `ros2 topic hz /sensors/camera_0/color/compressed` |
| Two different Pythons (pyenv vs system) conflict | pyenv shadowing system Python for ROS | Always use `/usr/bin/python3 -m pip install --user ...` for ROS deps |
| Appending to old `labels.csv` refused | v0.7.3 collector detects missing `odom_stamp` column in existing file | Rename old CSV aside: `mv labels.csv labels_old.csv`, then re-launch collector |

---

## Quick Reference — Full Pipeline in Order

```bash
# 0. Source workspace (every terminal)
source ~/ros2/orchard_navigation_rl_ws/install/setup.bash

# 1. Collect  (Terminal A: launch | Terminal B: toggle)
ros2 launch orchard_bc_training bc_collect.launch.py
ros2 service call /bc_data_collector/toggle_recording std_srvs/srv/SetBool "{data: true}"
# ... drive ...
ros2 service call /bc_data_collector/toggle_recording std_srvs/srv/SetBool "{data: false}"

# 2. Build cache
ros2 run orchard_bc_training build_cache \
    --data_root ~/ros2/orchard_navigation_rl_ws/data/raw

# 3. Train
ros2 run orchard_bc_training train \
    --data_root ~/ros2/orchard_navigation_rl_ws/data/raw \
    --out_dir   ~/ros2/orchard_navigation_rl_ws/checkpoints \
    --epochs 50 --batch_size 64 --lr 3e-4

# 4. Deploy  (Terminal A: launch | Terminal B: visualize)
ros2 launch orchard_bc_training bc_deploy.launch.py
ros2 run rqt_image_view rqt_image_view /bc_viz/image
```
