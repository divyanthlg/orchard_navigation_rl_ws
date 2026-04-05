"""
Data Collector Node — Behaviour Cloning
=========================================
Subscribes to camera + teleop cmd_vel from the Warthog and continuously
saves synchronized (image, velocity) pairs for BC training.

Recording starts immediately on launch and stops cleanly on Ctrl+C.

Topics (Warthog w200_0100):
    /w200_0100/sensors/camera_0/color/image   (sensor_msgs/Image)
    /w200_0100/rc_teleop/cmd_vel              (geometry_msgs/Twist)
"""

import os
import csv
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
    qos_profile_sensor_data)
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import cv2
from cv_bridge import CvBridge


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter(
            'image_topic', '/w200_0100/sensors/camera_0/color/image')
        self.declare_parameter(
            'cmd_vel_topic', '/w200_0100/rc_teleop/cmd_vel')
        self.declare_parameter(
            'image_dir',
            '/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/data/raw/images')
        self.declare_parameter(
            'labels_file',
            '/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/data/raw/labels.csv')
        self.declare_parameter('save_rate_hz', 5.0)
        self.declare_parameter('image_width', 256)
        self.declare_parameter('image_height', 256)
        self.declare_parameter('image_stale_sec', 0.5)
        self.declare_parameter('cmd_vel_stale_sec', 2.0)
        self.declare_parameter('min_linear_vel', 0.01)
        self.declare_parameter('skip_stationary', True)

        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.image_dir = self.get_parameter('image_dir').value
        self.labels_file = self.get_parameter('labels_file').value
        self.save_rate_hz = self.get_parameter('save_rate_hz').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.image_stale_sec = self.get_parameter('image_stale_sec').value
        self.cmd_vel_stale_sec = self.get_parameter('cmd_vel_stale_sec').value
        self.min_linear_vel = self.get_parameter('min_linear_vel').value
        self.skip_stationary = self.get_parameter('skip_stationary').value

        # ── State ───────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_image = None
        self.latest_image_time = None
        self.latest_cmd_vel = None
        self.latest_cmd_vel_time = None

        self.session_saved = 0
        self.session_skipped = 0

        # ── Setup directories & CSV ─────────────────────────────────────
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.labels_file), exist_ok=True)

        self.frame_counter = self._get_next_frame_number()

        # Create CSV with header if it doesn't exist
        if not os.path.isfile(self.labels_file):
            with open(self.labels_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'linear_vel', 'angular_vel'])
            self.get_logger().info(f'Created new labels file: {self.labels_file}')
        else:
            self.get_logger().info(f'Appending to existing labels: {self.labels_file}')

        # ── QoS for sensor data ─────────────────────────────────────────
        # ── Subscribers ─────────────────────────────────────────────────
        # Use rclpy's built-in sensor data profile for the image (BEST_EFFORT,
        # KEEP_LAST, depth=5) — this matches most ROS 2 camera drivers.
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, qos_profile_sensor_data)

        # cmd_vel from this teleop publishes BEST_EFFORT, use a matching profile
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self._cmd_vel_cb, cmd_qos)

        # ── Save timer ──────────────────────────────────────────────────
        self.save_timer = self.create_timer(
            1.0 / self.save_rate_hz, self._save_tick)

        # ── Startup banner ─────────────────────────────────────────────
        self.get_logger().info('')
        self.get_logger().info('╔══════════════════════════════════════════════════╗')
        self.get_logger().info('║       Orchard BC Data Collector — RECORDING     ║')
        self.get_logger().info('╠══════════════════════════════════════════════════╣')
        self.get_logger().info(f'║  Image topic:  {self.image_topic}')
        self.get_logger().info(f'║  Cmd topic:    {self.cmd_vel_topic}')
        self.get_logger().info(f'║  Save rate:    {self.save_rate_hz:.1f} Hz')
        self.get_logger().info(f'║  Image size:   {self.image_width}x{self.image_height}')
        self.get_logger().info(f'║  Next frame:   {self.frame_counter}')
        self.get_logger().info(f'║  Image dir:    {self.image_dir}')
        self.get_logger().info(f'║  Labels:       {self.labels_file}')
        self.get_logger().info('╠══════════════════════════════════════════════════╣')
        self.get_logger().info('║       Press Ctrl+C to stop recording             ║')
        self.get_logger().info('╚══════════════════════════════════════════════════╝')
        self.get_logger().info('')

    # ── Callbacks ───────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        with self.lock:
            self.latest_image = msg
            self.latest_image_time = self.get_clock().now().nanoseconds / 1e9

    def _cmd_vel_cb(self, msg: Twist):
        with self.lock:
            self.latest_cmd_vel = msg
            self.latest_cmd_vel_time = self.get_clock().now().nanoseconds / 1e9

    def _save_tick(self):
        """Called at save_rate_hz. Pairs latest image + cmd_vel and saves."""
        with self.lock:
            image_msg = self.latest_image
            image_time = self.latest_image_time
            cmd_msg = self.latest_cmd_vel
            cmd_time = self.latest_cmd_vel_time

        # Wait until both topics have data
        # Wait until both topics have data
        if image_msg is None:
            self.get_logger().warn(
                f'No image yet on {self.image_topic}',
                throttle_duration_sec=3.0)
            return
        if cmd_msg is None:
            self.get_logger().warn(
                f'No cmd_vel yet on {self.cmd_vel_topic}',
                throttle_duration_sec=3.0)
            return

        # Freshness check: both the image and the cmd_vel must have been
        # received recently. The RC teleop may only publish when the operator
        # is actively commanding, so we accept the last known cmd_vel as long
        # as it's not too stale (cmd_vel_stale_sec).
        now = self.get_clock().now().nanoseconds / 1e9
        image_age = now - image_time
        cmd_age = now - cmd_time

        if image_age > self.image_stale_sec:
            self.get_logger().warn(
                f'Image is stale: {image_age:.2f}s old',
                throttle_duration_sec=2.0)
            return
        if cmd_age > self.cmd_vel_stale_sec:
            self.get_logger().warn(
                f'cmd_vel is stale: {cmd_age:.2f}s old '
                f'(is the operator commanding the robot?)',
                throttle_duration_sec=2.0)
            return

        # Extract velocities
        linear_vel = cmd_msg.linear.x
        angular_vel = cmd_msg.angular.z

        # Skip stationary frames (robot not really moving)
        if self.skip_stationary and abs(linear_vel) < self.min_linear_vel:
            self.session_skipped += 1
            return

        # Convert image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        # Resize
        if self.image_width > 0 and self.image_height > 0:
            cv_image = cv2.resize(
                cv_image, (self.image_width, self.image_height),
                interpolation=cv2.INTER_LANCZOS4)

        # Save image
        filename = f'frame_{self.frame_counter:06d}.png'
        filepath = os.path.join(self.image_dir, filename)
        cv2.imwrite(filepath, cv_image)

        # Append to CSV
        with open(self.labels_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, f'{linear_vel:.6f}', f'{angular_vel:.6f}'])

        self.frame_counter += 1
        self.session_saved += 1

        # Periodic progress log
        if self.session_saved % 50 == 0:
            self.get_logger().info(
                f'Saved {self.session_saved} frames this session '
                f'(total: {self.frame_counter}, skipped: {self.session_skipped})')

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_next_frame_number(self) -> int:
        """Continue frame numbering from existing files."""
        if not os.path.isdir(self.image_dir):
            return 0
        existing = [f for f in os.listdir(self.image_dir)
                     if f.startswith('frame_') and f.endswith('.png')]
        if not existing:
            return 0
        numbers = []
        for f in existing:
            try:
                num = int(f.replace('frame_', '').replace('.png', ''))
                numbers.append(num)
            except ValueError:
                continue
        return max(numbers) + 1 if numbers else 0


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Print final summary
        node.get_logger().info('')
        node.get_logger().info('╔══════════════════════════════════════════════════╗')
        node.get_logger().info('║           Recording Stopped — Summary            ║')
        node.get_logger().info('╠══════════════════════════════════════════════════╣')
        node.get_logger().info(f'║  Saved this session:  {node.session_saved}')
        node.get_logger().info(f'║  Skipped (stationary): {node.session_skipped}')
        node.get_logger().info(f'║  Total frames on disk: {node.frame_counter}')
        node.get_logger().info('╚══════════════════════════════════════════════════╝')
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
