"""
bc_dataset_viewer.py — Scrub through your collected dataset and see each image
with predicted vs ground-truth velocity bars overlaid, exactly like bc_viz_node.

Usage:
    python3 bc_dataset_viewer.py \
        --data_root  ~/ros2/orchard_navigation_rl_ws/data/raw \
        --checkpoint ~/ros2/orchard_navigation_rl_ws/checkpoints/best.pt

Controls:
    SPACE / →     next frame
    ←             previous frame
    F             toggle auto-play
    +/-           faster / slower auto-play
    S             save current frame as PNG
    Q / ESC       quit

Bars (bottom of image):
    Top bar     ANG  — blue marker = model prediction, green marker = ground truth
    Bottom bar  LIN  — blue marker = model prediction, green marker = ground truth

Top-left panel shows numeric values for both.
"""

import argparse
import os
import sys
import csv
import time

import cv2
import numpy as np
import torch

# ── make the installed package importable if workspace is sourced ──────────
WS = os.path.expanduser('~/ros2/orchard_navigation_rl_ws')
for _sub in (
    'install/orchard_bc_training/lib/python3/dist-packages',
    'install/orchard_bc_training/lib/python3.10/dist-packages',
    'install/orchard_bc_training/lib/python3.11/dist-packages',
):
    _c = os.path.join(WS, _sub)
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)

from orchard_bc_training.models import OrchardNavModel   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_index(data_root: str):
    rows = []
    with open(os.path.join(data_root, 'index.csv')) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    filenames = [r['filename']          for r in rows]
    stamps    = np.array([float(r['stamp'])       for r in rows], dtype=np.float64)
    lin       = np.array([float(r['linear_vel'])  for r in rows], dtype=np.float32)
    ang       = np.array([float(r['angular_vel']) for r in rows], dtype=np.float32)
    return filenames, stamps, lin, ang


def find_valid_starts(stamps, seq_len=13, nominal_dt=0.5, gap_factor=1.5):
    max_dt = nominal_dt * gap_factor
    dts    = np.diff(stamps)
    bad    = dts > max_dt
    N      = len(stamps)
    valid  = [i for i in range(1, N - seq_len + 1)
              if not bad[i: i + seq_len - 1].any()]
    return np.array(valid, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Inference (vectorised over all windows at once — fast even for ~10k frames)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_all_inference(latents, lin, ang, valid_starts, seq_len,
                      model, device, batch_size=256):
    """Return (N_windows, 2) float32 array of [lin_pred, ang_pred]."""
    N = len(valid_starts)
    preds = np.zeros((N, 2), dtype=np.float32)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        idxs = valid_starts[start:end]
        B    = len(idxs)

        # latents: (B, T, 4, 32, 32)
        lat_np = np.stack(
            [np.array(latents[i: i + seq_len], dtype=np.float32) for i in idxs]
        )
        lat = torch.from_numpy(lat_np).to(device)

        # extras: (B, T, 4)
        ext_np = np.stack([
            np.stack([
                lin[i:     i + seq_len],
                ang[i:     i + seq_len],
                lin[i - 1: i + seq_len - 1],
                ang[i - 1: i + seq_len - 1],
            ], axis=-1)
            for i in idxs
        ], axis=0).astype(np.float32)
        ext = torch.from_numpy(ext_np).to(device)

        actions = model(lat, ext, is_latents=True)   # (B, T, 2)
        preds[start:end] = actions[:, -1, :].cpu().numpy()

        done = end
        if done % (batch_size * 4) == 0 or done == N:
            print(f'  inference {done}/{N} windows …', end='\r', flush=True)

    print(f'  inference done — {N} windows.          ')
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers  (same style as bc_viz_node.py)
# ─────────────────────────────────────────────────────────────────────────────

# BGR colour palette
C_BLUE      = (255, 140,  40)   # prediction  (bright blue)
C_GREEN     = ( 60, 220,  60)   # ground truth (green)
C_WHITE     = (255, 255, 255)
C_DARK      = ( 18,  18,  18)
C_PANEL_BG  = ( 28,  28,  28)
C_LABEL     = (200, 200, 200)
C_GOLD      = ( 40, 210, 255)   # accent / headers

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD   = cv2.FONT_HERSHEY_DUPLEX


def _bar(img, cx, cy, bar_w, bar_h,
         pred_val, gt_val, max_abs, label_str):
    """Draw one horizontal velocity bar (matches bc_viz_node._draw_bar)."""
    x1, x2 = cx - bar_w // 2, cx + bar_w // 2
    y1, y2 = cy - bar_h // 2, cy + bar_h // 2

    # semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 - 6, y1 - 18), (x2 + 6, y2 + 6), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # bar frame + zero line
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)
    cv2.line(img, (cx, y1 - 2), (cx, y2 + 2), (100, 100, 100), 1)

    def marker(val, color, thickness):
        clipped = float(np.clip(val, -max_abs, max_abs))
        frac    = clipped / max_abs
        mx      = int(cx + frac * (bar_w // 2 - 3))
        cv2.line(img, (mx, y1 - 3), (mx, y2 + 3), color, thickness)

    marker(gt_val,   C_GREEN, 2)   # ground truth — green
    marker(pred_val, C_BLUE,  3)   # prediction   — blue

    cv2.putText(img, label_str, (x1, y1 - 5),
                FONT, 0.38, C_LABEL, 1, cv2.LINE_AA)


def _info_panel(img, window_idx, n_windows, frame_idx, n_frames,
                lin_pred, ang_pred, lin_gt, ang_gt,
                max_lin, max_ang, filename, auto_play, fps):
    """Top-left info overlay panel."""
    lines = [
        (f'WIN {window_idx + 1}/{n_windows}  FR {frame_idx}',  C_GOLD,  0.46, 1),
        (f'',                                                    C_LABEL, 0.36, 1),
        (f'PRED  lin {lin_pred:+.3f}  ang {ang_pred:+.3f}',    C_BLUE,  0.40, 1),
        (f'GT    lin {lin_gt:+.3f}   ang {ang_gt:+.3f}',       C_GREEN, 0.40, 1),
        (f'',                                                    C_LABEL, 0.36, 1),
        (f'{"AUTO" if auto_play else "STEP"}  {fps:.1f} fps',   C_WHITE, 0.36, 1),
        (f'{os.path.basename(filename)}',                       (100,100,100), 0.32, 1),
    ]
    pad_x, pad_y = 8, 8
    line_h = 18
    panel_h = len(lines) * line_h + pad_y * 2
    panel_w = 240

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), C_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (0, 0), (panel_w, panel_h), (60, 60, 60), 1)

    y = pad_y + line_h
    for text, color, scale, thick in lines:
        cv2.putText(img, text, (pad_x, y), FONT, scale, color, thick, cv2.LINE_AA)
        y += line_h


def _legend(img):
    """Small legend bottom-right."""
    h, w = img.shape[:2]
    items = [('█ PRED', C_BLUE), ('█ GT', C_GREEN)]
    x = w - 10
    y = h - 110
    for label, color in reversed(items):
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.38, 1)
        x -= tw + 8
        cv2.putText(img, label, (x, y), FONT, 0.38, color, 1, cv2.LINE_AA)


def _progress_bar(img, progress):
    """Thin progress bar across the very top of the image."""
    h, w = img.shape[:2]
    filled = int(w * progress)
    cv2.rectangle(img, (0, 0), (w, 3), (40, 40, 40), -1)
    cv2.rectangle(img, (0, 0), (filled, 3), C_GOLD, -1)


def _controls_hint(img):
    h, w = img.shape[:2]
    hints = 'SPC/→ next   ← prev   F auto-play   +/- speed   S save   Q quit'
    cv2.putText(img, hints, (8, h - 6), FONT, 0.30, (80, 80, 80), 1, cv2.LINE_AA)


def build_frame(raw_img, lin_pred, ang_pred, lin_gt, ang_gt,
                max_lin, max_ang,
                window_idx, n_windows, frame_idx, n_frames,
                filename, auto_play, fps,
                display_h=480):
    """Compose the final display frame from a raw image + metadata."""
    h_orig, w_orig = raw_img.shape[:2]
    scale  = display_h / h_orig
    disp_w = int(w_orig * scale)
    frame  = cv2.resize(raw_img, (disp_w, display_h), interpolation=cv2.INTER_LINEAR)

    h, w = frame.shape[:2]
    cx   = w // 2

    bar_w = int(w * 0.55)
    bar_h = 16

    # ── two velocity bars ──────────────────────────────────────────
    ang_cy = h - 72
    lin_cy = h - 38

    _bar(frame, cx, ang_cy, bar_w, bar_h,
         ang_pred, ang_gt, max_ang,
         f'ANG   pred={ang_pred:+.2f}  gt={ang_gt:+.2f}  (±{max_ang:.1f} r/s)')

    _bar(frame, cx, lin_cy, bar_w, bar_h,
         lin_pred, lin_gt, max_lin,
         f'LIN   pred={lin_pred:+.2f}  gt={lin_gt:+.2f}  (±{max_lin:.1f} m/s)')

    # ── overlays ───────────────────────────────────────────────────
    _info_panel(frame, window_idx, n_windows, frame_idx, n_frames,
                lin_pred, ang_pred, lin_gt, ang_gt,
                max_lin, max_ang, filename, auto_play, fps)
    _legend(frame)
    _progress_bar(frame, window_idx / max(1, n_windows - 1))
    _controls_hint(frame)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main viewer loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Offline dataset viewer: images + velocity bar overlay.')
    ap.add_argument('--data_root',       required=True)
    ap.add_argument('--checkpoint',      required=True)
    ap.add_argument('--seq_len',         type=int,   default=13)
    ap.add_argument('--max_linear_vel',  type=float, default=1.0)
    ap.add_argument('--max_angular_vel', type=float, default=0.5)
    ap.add_argument('--display_height',  type=int,   default=520,
                    help='Resize image height for display (default 520)')
    ap.add_argument('--fps',             type=float, default=4.0,
                    help='Auto-play speed in frames/sec (default 4)')
    ap.add_argument('--start',           type=int,   default=0,
                    help='Start at this window index (default 0)')
    ap.add_argument('--save_dir',        default='./viewer_saves',
                    help='Directory for S-key screenshot saves')
    args = ap.parse_args()

    data_root = os.path.expanduser(args.data_root)
    image_dir = os.path.join(data_root, 'images')

    # ── load dataset metadata ─────────────────────────────────────
    print('Loading index …')
    filenames, stamps, lin, ang = load_index(data_root)
    valid_starts = find_valid_starts(stamps, seq_len=args.seq_len)
    n_windows    = len(valid_starts)
    print(f'  {n_windows} valid windows  ({len(stamps)} frames)')
    if n_windows == 0:
        print('ERROR: no valid windows — check --seq_len and data length.')
        sys.exit(1)

    # ── load latents ──────────────────────────────────────────────
    latents_path = os.path.join(data_root, 'latents.npy')
    print(f'Loading latents from {latents_path} …')
    latents = np.load(latents_path, mmap_mode='r')

    # ── load model ────────────────────────────────────────────────
    checkpoint = os.path.expanduser(args.checkpoint)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model from {checkpoint} on {device} …')
    model = OrchardNavModel(load_vae=False, seq_len=args.seq_len).to(device)
    model.load_trainable(checkpoint, map_location=device)
    model.eval()

    # ── run all inference up-front (fast — no VAE, just GRU + MLP) ─
    print('Running inference on all windows …')
    all_preds = run_all_inference(
        latents, lin, ang, valid_starts, args.seq_len, model, device)
    # clip + scale to real-world velocities
    all_preds[:, 0] = np.clip(all_preds[:, 0], -1.0, 1.0) * args.max_linear_vel
    all_preds[:, 1] = np.clip(all_preds[:, 1], -1.0, 1.0) * args.max_angular_vel

    # ── OpenCV window ─────────────────────────────────────────────
    WIN = 'BC Dataset Viewer  [SPACE/→ next | ← prev | F auto | +/- speed | S save | Q quit]'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, args.display_height + 20)

    window_idx = max(0, min(args.start, n_windows - 1))
    auto_play  = False
    fps        = args.fps
    last_auto  = time.time()
    os.makedirs(args.save_dir, exist_ok=True)

    # track display fps
    t_prev = time.time()

    while True:
        i         = int(valid_starts[window_idx])
        frame_idx = i + args.seq_len - 1          # last frame of the window

        # ── load image ─────────────────────────────────────────────
        fname   = filenames[frame_idx]
        imgpath = os.path.join(image_dir, fname)
        raw     = cv2.imread(imgpath)
        if raw is None:
            raw = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.putText(raw, f'NOT FOUND: {fname}', (10, 128),
                        FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # ── ground truth and prediction for this window ────────────
        lin_gt   = float(lin[frame_idx])
        ang_gt   = float(ang[frame_idx])
        lin_pred = float(all_preds[window_idx, 0])
        ang_pred = float(all_preds[window_idx, 1])

        # ── display fps ────────────────────────────────────────────
        t_now = time.time()
        disp_fps = 1.0 / max(1e-6, t_now - t_prev)
        t_prev = t_now

        # ── compose and show ───────────────────────────────────────
        display = build_frame(
            raw, lin_pred, ang_pred, lin_gt, ang_gt,
            args.max_linear_vel, args.max_angular_vel,
            window_idx, n_windows, frame_idx, len(stamps),
            fname, auto_play, disp_fps,
            display_h=args.display_height,
        )
        cv2.imshow(WIN, display)

        # ── auto-play delay ────────────────────────────────────────
        wait_ms = max(1, int(1000 / fps)) if auto_play else 50

        key = cv2.waitKey(wait_ms) & 0xFF

        # ── keyboard handling ──────────────────────────────────────
        if key in (ord('q'), 27):                    # Q / ESC — quit
            break

        elif key in (ord(' '), 83, 0xFF & ord('d')): # SPACE / → — next
            window_idx = min(window_idx + 1, n_windows - 1)
            auto_play  = False

        elif key in (81, 0xFF & ord('a')):            # ← — prev
            window_idx = max(window_idx - 1, 0)
            auto_play  = False

        elif key == ord('f'):                         # F — toggle auto
            auto_play = not auto_play
            last_auto = time.time()

        elif key == ord('+') or key == ord('='):      # + — faster
            fps = min(fps * 1.5, 60.0)

        elif key == ord('-'):                         # - — slower
            fps = max(fps / 1.5, 0.25)

        elif key == ord('s'):                         # S — save frame
            save_path = os.path.join(
                args.save_dir, f'win_{window_idx:05d}_fr_{frame_idx:06d}.png')
            cv2.imwrite(save_path, display)
            print(f'Saved: {save_path}')

        elif key == ord('g'):                         # G — go to frame (console)
            try:
                val = int(input('\nGo to window index: '))
                window_idx = max(0, min(val, n_windows - 1))
            except (ValueError, EOFError):
                pass

        # ── auto-advance ───────────────────────────────────────────
        if auto_play:
            now = time.time()
            if (now - last_auto) >= (1.0 / fps):
                window_idx = (window_idx + 1) % n_windows
                last_auto  = now

    cv2.destroyAllWindows()
    print('Viewer closed.')


if __name__ == '__main__':
    main()
