"""
DAgger Supervisor Node — v0.4 (Laptop Edition)
================================================
Logs (image, odom velocity) pairs during RC corrections.
Uses odom/filtered for ground-truth velocities (same as data collector).
Uses message_filters.ApproximateTimeSynchronizer for topic sync.
Data is saved to the LAPTOP's data/dagger_iterN/ folder.
"""

import os
import csv
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger

import message_filters
import cv2
from cv_bridge import CvBridge


class DAggerSupervisorNode(Node):
    def __init__(self):
        super().__init__('dagger_supervisor')

        self.declare_parameter('image_topic', '/w200_0100/sensors/camera_0/color/image')
        self.declare_parameter('odom_topic', '/w200_0100/platform/odom/filtered')
        self.declare_parameter('dagger_iteration', 1)
        self.declare_parameter('base_data_dir',
            '/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/data')
        self.declare_parameter('save_rate_hz', 5.0)
        self.declare_parameter('image_width', 256)
        self.declare_parameter('image_height', 256)
        self.declare_parameter('sync_slop_sec', 0.1)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('log_mode', 'corrections_only')

        image_topic     = self.get_parameter('image_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        dagger_iter     = self.get_parameter('dagger_iteration').value
        base_data_dir   = self.get_parameter('base_data_dir').value
        self.save_rate  = self.get_parameter('save_rate_hz').value
        self.image_width  = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.sync_slop    = self.get_parameter('sync_slop_sec').value
        self.sync_queue_size = self.get_parameter('sync_queue_size').value
        self.log_mode   = self.get_parameter('log_mode').value

        self.dagger_dir = os.path.join(base_data_dir, f'dagger_iter{dagger_iter}')
        self.image_dir = os.path.join(self.dagger_dir, 'images')
        self.labels_file = os.path.join(self.dagger_dir, 'labels.csv')
        os.makedirs(self.image_dir, exist_ok=True)

        if not os.path.isfile(self.labels_file):
            with open(self.labels_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'linear_vel', 'angular_vel', 'source'])

        self.bridge = CvBridge()
        self.active_source = 'policy'
        self.save_interval = 1.0 / self.save_rate
        self.last_save_time = 0.0

        self.frame_counter = self._get_next_frame_number()
        self.corrections_saved = 0
        self.policy_saved = 0
        self.total_human_interventions = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # ── message_filters subscribers ─────────────────────────────────
        self.image_sub = message_filters.Subscriber(
            self, Image, image_topic, qos_profile=sensor_qos)
        self.odom_sub = message_filters.Subscriber(
            self, Odometry, self.odom_topic, qos_profile=sensor_qos)

        # Both Image and Odometry have headers
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.odom_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self._synced_cb)

        # ── Mux source subscriber (not synced — just state tracking) ────
        self.source_sub = self.create_subscription(
            String, '/mux/active_source', self._source_cb, 10)

        self.status_srv = self.create_service(
            Trigger, '~/status', self._status_cb)

        self.get_logger().info(
            f'\n'
            f'╔══════════════════════════════════════════════════╗\n'
            f'║    DAgger Supervisor — Iteration {dagger_iter}               ║\n'
            f'╠══════════════════════════════════════════════════╣\n'
            f'║  Log mode:     {self.log_mode}\n'
            f'║  Save rate:    {self.save_rate:.1f} Hz\n'
            f'║  Odom topic:   {self.odom_topic}\n'
            f'║  Sync slop:    {self.sync_slop:.3f}s  (queue={self.sync_queue_size})\n'
            f'║  Data dir:     {self.dagger_dir}\n'
            f'║  Next frame:   {self.frame_counter}\n'
            f'╚══════════════════════════════════════════════════╝')

    def _source_cb(self, msg):
        new_source = msg.data
        if new_source == 'human' and self.active_source == 'policy':
            self.total_human_interventions += 1
            self.get_logger().info(
                f'Correction #{self.total_human_interventions} started')
        elif new_source == 'policy' and self.active_source == 'human':
            self.get_logger().info('Correction ended — back to policy')
        self.active_source = new_source

    def _synced_cb(self, image_msg: Image, odom_msg: Odometry):
        """Called by ApproximateTimeSynchronizer with a matched pair."""

        # Rate-limit saves
        now = time.monotonic()
        if (now - self.last_save_time) < self.save_interval:
            return

        if self.log_mode == 'corrections_only':
            if self.active_source != 'human':
                return

        source = self.active_source

        # Read velocities from filtered odometry
        linear_vel = odom_msg.twist.twist.linear.x
        angular_vel = odom_msg.twist.twist.angular.z

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        if self.image_width > 0 and self.image_height > 0:
            cv_image = cv2.resize(cv_image, (self.image_width, self.image_height),
                                  interpolation=cv2.INTER_LANCZOS4)

        filename = f'frame_{self.frame_counter:06d}.png'
        cv2.imwrite(os.path.join(self.image_dir, filename), cv_image)

        with open(self.labels_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, f'{linear_vel:.6f}', f'{angular_vel:.6f}', source])

        self.frame_counter += 1
        self.last_save_time = now

        if source == 'human':
            self.corrections_saved += 1
        else:
            self.policy_saved += 1

        total = self.corrections_saved + self.policy_saved
        if total % 50 == 0:
            self.get_logger().info(
                f'DAgger: {self.corrections_saved} corrections, '
                f'{self.total_human_interventions} interventions')

    def _status_cb(self, request, response):
        response.success = True
        response.message = (
            f'Source: {self.active_source} | '
            f'Corrections: {self.corrections_saved} | '
            f'Interventions: {self.total_human_interventions} | '
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
    node = DAggerSupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.get_logger().info(
                f'DAgger ended — {node.corrections_saved} corrections')
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
