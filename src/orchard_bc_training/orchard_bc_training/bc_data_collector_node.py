"""
BC Data Collector Node — v0.7.4  (compressed image input, per-session folders)
===============================================================================
Separate from the v0.6 data_collector in orchard_data_collector.

Each time recording is STARTED a new session folder is created:

    <data_root>/sessions/session_<YYYYMMDD_HHMMSS>/
        images/
        labels.csv          ← columns: filename, stamp, odom_stamp,
                                        linear_vel, angular_vel

build_cache discovers all session_* folders automatically and merges them
before encoding — no manual path management needed.

- Node name:          bc_data_collector
- Service namespace:  /bc_data_collector/...
- Default rate:       2 Hz
- Default data dir:   ~/ros2/orchard_navigation_rl_ws/data/raw
- Image input:        sensor_msgs/CompressedImage (JPEG/PNG payload)
"""

import os
import csv
import time
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, Trigger

import message_filters
import cv2
import numpy as np


DEFAULT_DATA_DIR = os.path.expanduser(
    '~/ros2/orchard_navigation_rl_ws/data/raw')


class BCDataCollectorNode(Node):

    def __init__(self):
        super().__init__('bc_data_collector')

        self.declare_parameter('image_topic',
            '/sensors/camera_0/color/compressed')
        self.declare_parameter('odom_topic',
            '/platform/odom/filtered')
        # data_root is the only path arg now; session subfolders are created
        # automatically under <data_root>/sessions/
        self.declare_parameter('data_root', DEFAULT_DATA_DIR)
        self.declare_parameter('save_rate_hz', 2.0)
        self.declare_parameter('image_width', 256)
        self.declare_parameter('image_height', 256)
        self.declare_parameter('sync_slop_sec', 0.1)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('min_linear_vel', 0.01)
        self.declare_parameter('skip_stationary', False)
        self.declare_parameter('auto_start', False)

        self.image_topic     = self.get_parameter('image_topic').value
        self.odom_topic      = self.get_parameter('odom_topic').value
        self.data_root       = os.path.expanduser(
                                   self.get_parameter('data_root').value)
        self.save_rate_hz    = self.get_parameter('save_rate_hz').value
        self.image_width     = self.get_parameter('image_width').value
        self.image_height    = self.get_parameter('image_height').value
        self.sync_slop       = self.get_parameter('sync_slop_sec').value
        self.sync_queue_size = self.get_parameter('sync_queue_size').value
        self.min_linear_vel  = self.get_parameter('min_linear_vel').value
        self.skip_stationary = self.get_parameter('skip_stationary').value
        auto_start           = self.get_parameter('auto_start').value

        # Session state — populated fresh on each _start_session() call
        self.session_dir     = None   # full path to current session folder
        self.image_dir       = None
        self.labels_file     = None
        self.frame_counter   = 0      # resets to 0 each session
        self.session_saved   = 0
        self.session_skipped = 0

        self.recording      = False
        self.lock           = threading.Lock()
        self.save_interval  = 1.0 / self.save_rate_hz
        self.last_save_time = 0.0

        self.sessions_root = os.path.join(self.data_root, 'sessions')
        os.makedirs(self.sessions_root, exist_ok=True)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.image_sub = message_filters.Subscriber(
            self, CompressedImage, self.image_topic, qos_profile=sensor_qos)
        self.odom_sub = message_filters.Subscriber(
            self, Odometry, self.odom_topic, qos_profile=sensor_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.odom_sub],
            queue_size=self.sync_queue_size, slop=self.sync_slop,
        )
        self.sync.registerCallback(self._synced_cb)

        self.toggle_srv = self.create_service(
            SetBool, '~/toggle_recording', self._toggle_recording_cb)
        self.status_srv = self.create_service(
            Trigger, '~/status', self._status_cb)

        self.get_logger().info('')
        self.get_logger().info('╔══════════════════════════════════════════════════════╗')
        self.get_logger().info('║   BC Data Collector v0.7.4 — PAUSED                 ║')
        self.get_logger().info('╠══════════════════════════════════════════════════════╣')
        self.get_logger().info(f'║  Image topic:  {self.image_topic}')
        self.get_logger().info(f'║  Image type:   sensor_msgs/CompressedImage')
        self.get_logger().info(f'║  Odom topic:   {self.odom_topic}')
        self.get_logger().info(f'║  Save rate:    {self.save_rate_hz:.1f} Hz')
        self.get_logger().info(f'║  Image size:   {self.image_width}x{self.image_height}')
        self.get_logger().info(f'║  Sessions dir: {self.sessions_root}')
        self.get_logger().info('╠══════════════════════════════════════════════════════╣')
        self.get_logger().info('║  Toggle:                                             ║')
        self.get_logger().info('║   ros2 service call /bc_data_collector/\\             ║')
        self.get_logger().info('║     toggle_recording std_srvs/srv/SetBool \\          ║')
        self.get_logger().info('║     "{data: true}"                                   ║')
        self.get_logger().info('╚══════════════════════════════════════════════════════╝')

        if auto_start:
            self._start_session()

    # ── session lifecycle ──────────────────────────────────────────────

    def _start_session(self):
        """Create a timestamped session folder and initialise its labels.csv."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir   = os.path.join(self.sessions_root, f'session_{ts}')
        self.image_dir     = os.path.join(self.session_dir, 'images')
        self.labels_file   = os.path.join(self.session_dir, 'labels.csv')
        self.frame_counter  = 0
        self.session_saved  = 0
        self.session_skipped = 0
        self.last_save_time  = 0.0

        os.makedirs(self.image_dir, exist_ok=True)
        with open(self.labels_file, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['filename', 'stamp', 'odom_stamp', 'linear_vel', 'angular_vel'])

        self.recording = True
        self.get_logger().info(
            f'>>> RECORDING STARTED <<<\n'
            f'    Session folder: {self.session_dir}')

    def _stop_session(self):
        self.recording = False
        self.get_logger().info(
            f'>>> RECORDING STOPPED <<<\n'
            f'    {self.session_saved} frames saved → {self.session_dir}')

    # ── synced callback ────────────────────────────────────────────────

    @staticmethod
    def _decode_compressed(msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR

    def _synced_cb(self, image_msg: CompressedImage, odom_msg: Odometry):
        if not self.recording:
            return
        now = time.monotonic()
        if (now - self.last_save_time) < self.save_interval:
            return

        linear_vel  = odom_msg.twist.twist.linear.x
        angular_vel = odom_msg.twist.twist.angular.z

        if self.skip_stationary and abs(linear_vel) < self.min_linear_vel:
            self.session_skipped += 1
            return

        try:
            cv_image = self._decode_compressed(image_msg)
            if cv_image is None:
                raise RuntimeError('cv2.imdecode returned None')
        except Exception as e:
            self.get_logger().error(f'Image decode failed: {e}')
            return

        if self.image_width > 0 and self.image_height > 0:
            cv_image = cv2.resize(
                cv_image, (self.image_width, self.image_height),
                interpolation=cv2.INTER_LANCZOS4)

        stamp      = (image_msg.header.stamp.sec
                      + image_msg.header.stamp.nanosec * 1e-9)
        odom_stamp = (odom_msg.header.stamp.sec
                      + odom_msg.header.stamp.nanosec * 1e-9)

        filename = f'frame_{self.frame_counter:06d}.png'
        cv2.imwrite(os.path.join(self.image_dir, filename), cv_image)

        with open(self.labels_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                filename, f'{stamp:.6f}', f'{odom_stamp:.6f}',
                f'{linear_vel:.6f}', f'{angular_vel:.6f}',
            ])

        self.frame_counter  += 1
        self.session_saved  += 1
        self.last_save_time  = now

        if self.session_saved % 50 == 0:
            self.get_logger().info(
                f'Session: {self.session_saved} frames saved')

    # ── services ──────────────────────────────────────────────────────

    def _toggle_recording_cb(self, request, response):
        want_record = request.data
        if want_record and not self.recording:
            self._start_session()
        elif not want_record and self.recording:
            self._stop_session()
        # idempotent: already-recording + true, or already-stopped + false → no-op
        response.success = True
        response.message = f'Recording: {"ON" if self.recording else "OFF"}'
        return response

    def _status_cb(self, request, response):
        state = 'RECORDING' if self.recording else 'PAUSED'
        sess  = os.path.basename(self.session_dir) if self.session_dir else 'none'
        response.success = True
        response.message = (
            f'State: {state} | Session: {sess} | '
            f'Frames this session: {self.session_saved}')
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = BCDataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.get_logger().info('Shutdown.')
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
