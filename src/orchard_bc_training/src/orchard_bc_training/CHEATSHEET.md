# orchard_bc_training — Cheat Sheet (v0.7.3)

Workspace: `~/ros2/orchard_navigation_rl_ws`

**v0.7.3 change:** `labels.csv` now records both image and odom timestamps.
Columns: `filename, stamp, odom_stamp, linear_vel, angular_vel`.
`stamp` is the image header timestamp (primary, used for gap detection in
training); `odom_stamp` is the odom header timestamp (diagnostic — diff
against `stamp` to verify sync quality).

**v0.7.2 change:** image input is now `sensor_msgs/CompressedImage`.
Default topic: `/camera/camera/color/image_raw/compressed`.

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
`image_dir`, `labels_file`, `image_topic` (defaults to
`/camera/camera/color/image_raw/compressed`).

To override the topic if needed:
```bash
ros2 launch orchard_bc_training bc_collect.launch.py \
    image_topic:=/your/compressed/topic
```

Multiple sessions append to the same CSV. Session gaps are detected at
dataset-build time via the `stamp` column, so no training window crosses a
gap.

**Sanity check your input topic is CompressedImage:**
```bash
ros2 topic info /camera/camera/color/image_raw/compressed
# Type should be: sensor_msgs/msg/CompressedImage
```

---

## Step 2 — Build cache

```bash
ros2 run orchard_bc_training build_cache \
    --data_root ~/ros2/orchard_navigation_rl_ws/data/raw
```

Passes every saved PNG through the frozen VAE once, saves `latents.npy` (float16)
+ `index.csv`. Re-run whenever `labels.csv` grows. (This step is unchanged by
the compressed-topic switch — the collector has already decoded + saved PNGs
to disk.)

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
| `bc_policy_node` | Subscribes to compressed image, decodes + VAE encode @ 2 Hz, publishes `/bc_policy/cmd_vel` @ 10 Hz |
| `bc_cmd_vel_mux` | Forwards policy → `/w200_0100/cmd_vel`, RC overrides instantly |
| `bc_status_display` | Terminal indicator: `[POLICY]` / `[HUMAN]` + numeric cmd |
| `bc_viz_node` | Subscribes to compressed image, overlays command bars, publishes raw `Image` on `/bc_viz/image` |

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
    vae_model_id:=$HOME/other_vae_snapshot \
    image_topic:=/some/other/compressed/topic
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Hangs at `Loading model on cuda...` | No internet + no local VAE. Pre-download (Option A) or copy snapshot to `models/sd-vae-ft-mse/` (Option B). |
| `cv2.imdecode returned None` on every frame | Topic is publishing non-JPEG/PNG data or zero-byte payloads. `ros2 topic echo <topic> --no-arr` to inspect `.format`. |
| No frames arriving | Topic mismatch. `ros2 topic hz /camera/camera/color/image_raw/compressed` — if 0 Hz, check the actual topic name or whether the camera driver is publishing the compressed variant. |
| `PIL.Image has no attribute 'Resampling'` | `/usr/bin/python3 -m pip install --user 'Pillow>=10'` |
| `No such file: best.pt` | Override `checkpoint_path:=...` or move checkpoint. |
| `No such file: latents.npy` | Re-run step 2 with matching `--data_root`. |
| `0 valid windows` during train | All sessions shorter than 6.5 s. Record longer continuous runs. |
| Viz image blank / not updating | `ros2 topic hz /bc_viz/image` — 0 Hz means the compressed stream isn't arriving or is failing to decode. |
| Policy keeps publishing zeros | Buffer not filling. `ros2 topic hz <image_topic>`. |
| Two different Pythons (pyenv vs system) | Always use `/usr/bin/python3 -m pip install --user ...` for ROS deps. |

---

## Topics / services reference

**Published:** `/bc_policy/cmd_vel`, `/bc_policy/active`, `/w200_0100/cmd_vel`,
`/bc_mux/active_source`, `/bc_viz/image` (raw Image).

**Subscribed:** `/camera/camera/color/image_raw/compressed` (CompressedImage),
`/w200_0100/platform/odom/filtered`, `/w200_0100/rc_teleop/cmd_vel`.

**Services:** `/bc_data_collector/toggle_recording`,
`/bc_data_collector/status`.

---

## Migration notes (v0.7 → v0.7.3)

- `labels.csv` gained a column: `odom_stamp`. New files written by
  v0.7.3 have header `filename, stamp, odom_stamp, linear_vel, angular_vel`.
  Pre-v0.7.3 CSVs (4 columns) are **still readable by `build_cache`** — it
  fills `odom_stamp = stamp` for back-compat. **Do not append v0.7.3 data
  to a pre-v0.7.3 `labels.csv`**; the collector will refuse and tell you to
  rename the old file or migrate it first.
- `index.csv` written by `build_cache` gained the same `odom_stamp` column.
  `datasets.py` still uses only `stamp`, so no retraining is forced by
  this change.
- Image type: `sensor_msgs/Image` → `sensor_msgs/CompressedImage` in
  `bc_data_collector_node`, `bc_policy_node`, `bc_viz_node`.
- Decoding: `cv_bridge.imgmsg_to_cv2(...)` replaced with
  `cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)`.
  The policy node converts BGR→RGB after decoding (the torch transform
  expects RGB, same as v0.7).
- Default `image_topic` in both launch files updated.
- `/bc_viz/image` is still published as a raw `sensor_msgs/Image` (annotated),
  so `rqt_image_view` works with no changes on your side.
- `build_cache.py` is updated (new column); `train.py`, `datasets.py`,
  `models.py`, `vae_resolve.py`, `bc_cmd_vel_mux_node.py`,
  `bc_status_display.py` are unchanged.
- Existing PNGs + pre-v0.7.3 `labels.csv` remain fully compatible with
  v0.7.3's `build_cache` and training — no re-collection needed just to
  upgrade.

### Quick sync-quality check after collection

```python
# from the data/raw/ directory
import pandas as pd
df = pd.read_csv('labels.csv')
if 'odom_stamp' in df:
    diff_ms = (df['stamp'] - df['odom_stamp']) * 1000
    print(diff_ms.describe())
    # Expect |mean| and std well under 100 ms (your sync_slop_sec default).
    # If p99 > 100 ms, loosen sync_slop or check topic QoS.
```
