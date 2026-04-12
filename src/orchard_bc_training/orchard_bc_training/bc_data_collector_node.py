"""
BC Data Collector Node — v0.7  (in orchard_bc_training)
=========================================================
Separate from the v0.6 data_collector in orchard_data_collector.
- Node name:          bc_data_collector
- Service namespace:  /bc_data_collector/...
- CSV columns:        filename, stamp, linear_vel, angular_vel
- Default rate:       2 Hz
- Default data dir:   ~/ros2/orchard_navigation_rl_ws/data/raw
"""

import os
import csv
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, Trigger

import message_filters
import cv2
from cv_bridge import CvBridge


DEFAULT_DATA_DIR = os.path.expanduser(
    '~/ros2/orchard_navigation_rl_ws/data/raw')


class BCDataCollectorNode(Node):

    def __init__(self):
        super().__init__('bc_data_collector')

        self.declare_parameter('image_topic',
            '/w200_0100/sensors/camera_0/color/image')
        self.declare_parameter('odom_topic',
            '/w200_0100/platform/odom/filtered')
        self.declare_parameter('image_dir',
            os.path.join(DEFAULT_DATA_DIR, 'images'))
        self.declare_parameter('labels_file',
            os.path.join(DEFAULT_DATA_DIR, 'labels.csv'))
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
        self.image_dir       = self.get_parameter('image_dir').value
        self.labels_file     = self.get_parameter('labels_file').value
        self.save_rate_hz    = self.get_parameter('save_rate_hz').value
        self.image_width     = self.get_parameter('image_width').value
        self.image_height    = self.get_parameter('image_height').value
        self.sync_slop       = self.get_parameter('sync_slop_sec').value
        self.sync_queue_size = self.get_parameter('sync_queue_size').value
        self.min_linear_vel  = self.get_parameter('min_linear_vel').value
        self.skip_stationary = self.get_parameter('skip_stationary').value
        auto_start           = self.get_parameter('auto_start').value

        self.bridge = CvBridge()
        self.recording = auto_start
        self.lock = threading.Lock()
        self.save_interval = 1.0 / self.save_rate_hz
        self.last_save_time = 0.0
        self.session_saved = 0
        self.session_skipped = 0

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.labels_file), exist_ok=True)
        self.frame_counter = self._get_next_frame_number()

        if not os.path.isfile(self.labels_file):
            with open(self.labels_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'stamp', 'linear_vel', 'angular_vel'])
            self.get_logger().info(f'Created new labels file: {self.labels_file}')
        else:
            self.get_logger().info(f'Appending to existing labels: {self.labels_file}')
            self._verify_stamp_column()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.image_sub = message_filters.Subscriber(
            self, Image, self.image_topic, qos_profile=sensor_qos)
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

        state = 'RECORDING' if self.recording else 'PAUSED'
        self.get_logger().info('')
        self.get_logger().info('╔══════════════════════════════════════════════════╗')
        self.get_logger().info(f'║   BC Data Collector v0.7 — {state:<20s} ║')
        self.get_logger().info('╠══════════════════════════════════════════════════╣')
        self.get_logger().info(f'║  Image topic:  {self.image_topic}')
        self.get_logger().info(f'║  Odom topic:   {self.odom_topic}')
        self.get_logger().info(f'║  Save rate:    {self.save_rate_hz:.1f} Hz')
        self.get_logger().info(f'║  Image size:   {self.image_width}x{self.image_height}')
        self.get_logger().info(f'║  Next frame:   {self.frame_counter}')
        self.get_logger().info(f'║  Image dir:    {self.image_dir}')
        self.get_logger().info(f'║  Labels:       {self.labels_file}')
        self.get_logger().info('╠══════════════════════════════════════════════════╣')
        self.get_logger().info('║  Toggle:                                         ║')
        self.get_logger().info('║   ros2 service call /bc_data_collector/\\         ║')
        self.get_logger().info('║     toggle_recording std_srvs/srv/SetBool \\      ║')
        self.get_logger().info('║     "{data: true}"                               ║')
        self.get_logger().info('╚══════════════════════════════════════════════════╝')

    def _verify_stamp_column(self):
        with open(self.labels_file, 'r') as f:
            header = f.readline().strip().split(',')
        if 'stamp' not in header:
            self.get_logger().error(
                'EXISTING labels.csv has no "stamp" column — move it aside '
                'and start fresh, otherwise training will break.')

    def _synced_cb(self, image_msg: Image, odom_msg: Odometry):
        if not self.recording:
            return
        now = time.monotonic()
        if (now - self.last_save_time) < self.save_interval:
            return

        linear_vel = odom_msg.twist.twist.linear.x
        angular_vel = odom_msg.twist.twist.angular.z

        if self.skip_stationary and abs(linear_vel) < self.min_linear_vel:
            self.session_skipped += 1
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        if self.image_width > 0 and self.image_height > 0:
            cv_image = cv2.resize(
                cv_image, (self.image_width, self.image_height),
                interpolation=cv2.INTER_LANCZOS4)

        stamp = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9

        filename = f'frame_{self.frame_counter:06d}.png'
        filepath = os.path.join(self.image_dir, filename)
        cv2.imwrite(filepath, cv_image)

        with open(self.labels_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename, f'{stamp:.6f}',
                f'{linear_vel:.6f}', f'{angular_vel:.6f}',
            ])

        self.frame_counter += 1
        self.session_saved += 1
        self.last_save_time = now

        if self.session_saved % 50 == 0:
            self.get_logger().info(
                f'Session: {self.session_saved} saved, '
                f'total frames: {self.frame_counter}')

    def _toggle_recording_cb(self, request, response):
        was_recording = self.recording
        self.recording = request.data
        if self.recording and not was_recording:
            self.get_logger().info('>>> RECORDING STARTED <<<')
        elif not self.recording and was_recording:
            self.get_logger().info(
                f'>>> RECORDING STOPPED <<< (session: {self.session_saved} saved)')
        response.success = True
        response.message = f'Recording: {"ON" if self.recording else "OFF"}'
        return response

    def _status_cb(self, request, response):
        state = 'RECORDING' if self.recording else 'PAUSED'
        response.success = True
        response.message = (
            f'State: {state} | Session: {self.session_saved} | '
            f'Total: {self.frame_counter}')
        self.get_logger().info(response.message)
        return response

    def _get_next_frame_number(self) -> int:
        if not os.path.isdir(self.image_dir):
            return 0
        existing = [f for f in os.listdir(self.image_dir)
                    if f.startswith('frame_') and f.endswith('.png')]
        if not existing:
            return 0
        numbers = []
        for f in existing:
            try:
                numbers.append(int(f.replace('frame_', '').replace('.png', '')))
            except ValueError:
                continue
        return max(numbers) + 1 if numbers else 0


def main(args=None):
    rclpy.init(args=args)
    node = BCDataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.get_logger().info(
                f'Shutdown — total frames: {node.frame_counter}')
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
