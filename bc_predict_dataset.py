"""
bc_predict_dataset.py — Run inference on a collected dataset and publish
predicted vs. ground-truth control values for comparison in rqt_plot.

Usage:
    # Terminal A — source your workspace, then:
    python3 bc_predict_dataset.py \
        --data_root ~/ros2/orchard_navigation_rl_ws/data/raw \
        --checkpoint ~/ros2/orchard_navigation_rl_ws/checkpoints/best.pt \
        --seq_len 13 \
        --rate_hz 10.0

    # Terminal B — visualise in rqt_plot:
    ros2 run rqt_plot rqt_plot \
        /bc_pred/cmd_vel/linear/x \
        /bc_pred/cmd_vel/angular/z \
        /bc_gt/cmd_vel/linear/x \
        /bc_gt/cmd_vel/angular/z

Topics published
────────────────
  /bc_pred/cmd_vel   (geometry_msgs/Twist)  — model prediction
  /bc_gt/cmd_vel     (geometry_msgs/Twist)  — ground-truth label from CSV
  /bc_pred/progress  (std_msgs/Float32)     — 0.0-1.0 playback progress

The script loops through every valid 13-frame window in the dataset in order,
publishing at --rate_hz.  When it reaches the end it stops (or loops if you
pass --loop).
"""

import argparse
import os
import sys
import csv
import time

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

# ── make sure the installed package is importable ──────────────────────────
# If you source install/setup.bash before running this script, the next two
# lines are unnecessary — but they guard against a forgotten source.
WS = os.path.expanduser('~/ros2/orchard_navigation_rl_ws')
for sub in ('install/orchard_bc_training/lib/python3/dist-packages',
            'install/orchard_bc_training/lib/python3.10/dist-packages',
            'install/orchard_bc_training/lib/python3.11/dist-packages'):
    candidate = os.path.join(WS, sub)
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from orchard_bc_training.models import OrchardNavModel   # noqa: E402


# ── dataset helpers (mirrors datasets.py logic, no PyTorch Dataset needed) ─

def load_index(data_root: str):
    """Return stamps, lin, ang arrays from index.csv."""
    index_path = os.path.join(data_root, 'index.csv')
    stamps, lin, ang = [], [], []
    with open(index_path) as f:
        for row in csv.DictReader(f):
            stamps.append(float(row['stamp']))
            lin.append(float(row['linear_vel']))
            ang.append(float(row['angular_vel']))
    return (np.array(stamps, dtype=np.float64),
            np.array(lin,    dtype=np.float32),
            np.array(ang,    dtype=np.float32))


def find_valid_starts(stamps, seq_len=13, nominal_dt=0.5, gap_factor=1.5):
    """Replicate datasets.py gap-rejection logic."""
    max_dt = nominal_dt * gap_factor
    dts = np.diff(stamps)
    bad = dts > max_dt
    N = len(stamps)
    valid = []
    for i in range(1, N - seq_len + 1):
        if not bad[i: i + seq_len - 1].any():
            valid.append(i)
    return np.array(valid, dtype=np.int64)


# ── ROS node ───────────────────────────────────────────────────────────────

class DatasetReplayNode(Node):
    def __init__(self, args):
        super().__init__('bc_predict_dataset')

        self.pred_pub     = self.create_publisher(Twist,   '/bc_pred/cmd_vel',  10)
        self.gt_pub       = self.create_publisher(Twist,   '/bc_gt/cmd_vel',    10)
        self.progress_pub = self.create_publisher(Float32, '/bc_pred/progress', 10)

        # ── load dataset ──────────────────────────────────────────────
        data_root = os.path.expanduser(args.data_root)
        latents_path = os.path.join(data_root, 'latents.npy')

        self.get_logger().info(f'Loading latents from {latents_path} …')
        self.latents = np.load(latents_path, mmap_mode='r')   # float16, memory-mapped
        self.get_logger().info(
            f'  shape: {self.latents.shape}  dtype: {self.latents.dtype}')

        stamps, self.lin, self.ang = load_index(data_root)
        self.valid_starts = find_valid_starts(stamps, seq_len=args.seq_len)
        self.get_logger().info(
            f'  {len(self.valid_starts)} valid windows  '
            f'({len(stamps)} total frames)')

        if len(self.valid_starts) == 0:
            self.get_logger().error(
                'No valid windows found — check seq_len / data length.')
            raise SystemExit(1)

        # ── load model ────────────────────────────────────────────────
        checkpoint = os.path.expanduser(args.checkpoint)
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        self.get_logger().info(
            f'Loading checkpoint {checkpoint} on {device_str} …')

        self.model = OrchardNavModel(
            load_vae=False,           # latents already on disk — no VAE needed
            seq_len=args.seq_len,
        ).to(self.device)
        self.model.load_trainable(checkpoint, map_location=self.device)
        self.model.eval()
        self.get_logger().info('Model ready.')

        # ── runtime state ─────────────────────────────────────────────
        self.seq_len   = args.seq_len
        self.max_lin   = args.max_linear_vel
        self.max_ang   = args.max_angular_vel
        self.loop      = args.loop
        self.window_idx = 0          # index into self.valid_starts

        period = 1.0 / args.rate_hz
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f'Publishing at {args.rate_hz:.1f} Hz  '
            f'(loop={self.loop})\n'
            f'  /bc_pred/cmd_vel  — model prediction\n'
            f'  /bc_gt/cmd_vel    — ground-truth label\n'
            f'  /bc_pred/progress — 0→1 playback progress\n\n'
            f'View in rqt_plot:\n'
            f'  ros2 run rqt_plot rqt_plot \\\n'
            f'    /bc_pred/cmd_vel/linear/x \\\n'
            f'    /bc_pred/cmd_vel/angular/z \\\n'
            f'    /bc_gt/cmd_vel/linear/x \\\n'
            f'    /bc_gt/cmd_vel/angular/z')

    # ── per-tick inference ─────────────────────────────────────────────────
    @torch.no_grad()
    def _tick(self):
        if self.window_idx >= len(self.valid_starts):
            if self.loop:
                self.get_logger().info('Dataset end — looping.')
                self.window_idx = 0
            else:
                self.get_logger().info('Dataset end — done.')
                self.timer.cancel()
                return

        i = int(self.valid_starts[self.window_idx])
        T = self.seq_len

        # ── build input tensors ──────────────────────────────────────
        # Latents: float16 on disk → float32 on device
        lat_np = np.array(self.latents[i: i + T], dtype=np.float32)
        lat = torch.from_numpy(lat_np).unsqueeze(0).to(self.device)  # (1,T,4,32,32)

        # Extras: [cur_lin, cur_ang, prev_label_lin, prev_label_ang]
        cur_lin = self.lin[i:     i + T]
        cur_ang = self.ang[i:     i + T]
        prv_lin = self.lin[i - 1: i + T - 1]
        prv_ang = self.ang[i - 1: i + T - 1]
        extras_np = np.stack([cur_lin, cur_ang, prv_lin, prv_ang], axis=-1)
        extras = torch.from_numpy(extras_np).unsqueeze(0).to(self.device)  # (1,T,4)

        # ── run model, take last timestep ───────────────────────────
        actions = self.model(lat, extras, is_latents=True)  # (1,T,2)
        a = actions[0, -1].cpu().numpy()

        lin_pred = float(np.clip(a[0], -1.0, 1.0)) * self.max_lin
        ang_pred = float(np.clip(a[1], -1.0, 1.0)) * self.max_ang

        # Ground-truth label at the LAST frame of the window
        lin_gt = float(self.lin[i + T - 1])
        ang_gt = float(self.ang[i + T - 1])

        # ── publish ──────────────────────────────────────────────────
        pred_msg = Twist()
        pred_msg.linear.x  = lin_pred
        pred_msg.angular.z = ang_pred
        self.pred_pub.publish(pred_msg)

        gt_msg = Twist()
        gt_msg.linear.x  = lin_gt
        gt_msg.angular.z = ang_gt
        self.gt_pub.publish(gt_msg)

        prog = Float32()
        prog.data = float(self.window_idx) / max(1, len(self.valid_starts) - 1)
        self.progress_pub.publish(prog)

        self.window_idx += 1

        # ── console summary every 100 steps ─────────────────────────
        if self.window_idx % 100 == 0:
            self.get_logger().info(
                f'[{self.window_idx:5d}/{len(self.valid_starts)}]  '
                f'pred lin={lin_pred:+.3f} ang={ang_pred:+.3f}  |  '
                f'gt   lin={lin_gt:+.3f}   ang={ang_gt:+.3f}')


# ── entry point ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Replay dataset through BC policy and publish predictions.')
    ap.add_argument('--data_root', required=True,
                    help='Path to data/raw (must contain latents.npy + index.csv)')
    ap.add_argument('--checkpoint', required=True,
                    help='Path to best.pt or last.pt')
    ap.add_argument('--seq_len',        type=int,   default=13)
    ap.add_argument('--rate_hz',        type=float, default=10.0,
                    help='Playback speed in Hz (default 10)')
    ap.add_argument('--max_linear_vel', type=float, default=1.0)
    ap.add_argument('--max_angular_vel',type=float, default=0.5)
    ap.add_argument('--loop',           action='store_true',
                    help='Loop back to the start when the dataset is exhausted')

    # rclpy passes its own args after --ros-args; argparse sees only ours
    import sys as _sys
    idx = _sys.argv.index('--ros-args') if '--ros-args' in _sys.argv else len(_sys.argv)
    args = ap.parse_args(_sys.argv[1:idx])

    rclpy.init()
    try:
        node = DatasetReplayNode(args)
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
