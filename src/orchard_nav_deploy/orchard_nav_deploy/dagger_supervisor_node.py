"""
DAgger Supervisor Node — v0.3 (Laptop Edition)
================================================
Logs (image, human_cmd_vel) pairs during RC corrections.
Data is saved to the LAPTOP's data/dagger_iterN/ folder.
"""

import os
import csv
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import Trigger

import cv2
from cv_bridge import CvBridge


class DAggerSupervisorNode(Node):
    def __init__(self):
        super().__init__('dagger_supervisor')

        self.declare_parameter('image_topic', '/w200_0100/sensors/camera_0/color/image')
        self.declare_parameter('active_cmd_topic', '/w200_0100/cmd_vel')
        self.declare_parameter('human_cmd_topic', '/w200_0100/rc_teleop/cmd_vel')
        self.declare_parameter('dagger_iteration', 1)
        self.declare_parameter('base_data_dir',
            '/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/data')
        self.declare_parameter('save_rate_hz', 5.0)
        self.declare_parameter('image_width', 256)
        self.declare_parameter('image_height', 256)
        self.declare_parameter('sync_tolerance_sec', 0.1)
        self.declare_parameter('log_mode', 'corrections_only')

        image_topic = self.get_parameter('image_topic').value
        active_cmd_topic = self.get_parameter('active_cmd_topic').value
        dagger_iter = self.get_parameter('dagger_iteration').value
        base_data_dir = self.get_parameter('base_data_dir').value
        self.save_rate = self.get_parameter('save_rate_hz').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.sync_tolerance = self.get_parameter('sync_tolerance_sec').value
        self.log_mode = self.get_parameter('log_mode').value

        self.dagger_dir = os.path.join(base_data_dir, f'dagger_iter{dagger_iter}')
        self.image_dir = os.path.join(self.dagger_dir, 'images')
        self.labels_file = os.path.join(self.dagger_dir, 'labels.csv')
        os.makedirs(self.image_dir, exist_ok=True)

        if not os.path.isfile(self.labels_file):
            with open(self.labels_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'linear_vel', 'angular_vel', 'source'])

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.active_source = 'policy'

        self.latest_image = None
        self.latest_image_time = None
        self.latest_active_cmd = None
        self.latest_active_cmd_time = None

        self.frame_counter = self._get_next_frame_number()
        self.corrections_saved = 0
        self.policy_saved = 0
        self.total_human_interventions = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.image_sub = self.create_subscription(
            Image, image_topic, self._image_cb, sensor_qos)
        self.active_cmd_sub = self.create_subscription(
            Twist, active_cmd_topic, self._active_cmd_cb, 10)
        self.source_sub = self.create_subscription(
            String, '/mux/active_source', self._source_cb, 10)

        self.status_srv = self.create_service(
            Trigger, '~/status', self._status_cb)

        self.save_timer = self.create_timer(1.0 / self.save_rate, self._save_tick)

        self.get_logger().info(
            f'\n'
            f'╔══════════════════════════════════════════════════╗\n'
            f'║    DAgger Supervisor — Iteration {dagger_iter}               ║\n'
            f'╠══════════════════════════════════════════════════╣\n'
            f'║  Log mode:     {self.log_mode}\n'
            f'║  Save rate:    {self.save_rate:.1f} Hz\n'
            f'║  Data dir:     {self.dagger_dir}\n'
            f'║  Next frame:   {self.frame_counter}\n'
            f'╚══════════════════════════════════════════════════╝')

    def _image_cb(self, msg):
        with self.lock:
            self.latest_image = msg
            self.latest_image_time = self.get_clock().now().nanoseconds / 1e9

    def _active_cmd_cb(self, msg):
        with self.lock:
            self.latest_active_cmd = msg
            self.latest_active_cmd_time = self.get_clock().now().nanoseconds / 1e9

    def _source_cb(self, msg):
        new_source = msg.data
        if new_source == 'human' and self.active_source == 'policy':
            self.total_human_interventions += 1
            self.get_logger().info(
                f'Correction #{self.total_human_interventions} started')
        elif new_source == 'policy' and self.active_source == 'human':
            self.get_logger().info('Correction ended — back to policy')
        self.active_source = new_source

    def _save_tick(self):
        with self.lock:
            image_msg = self.latest_image
            image_time = self.latest_image_time
            active_cmd = self.latest_active_cmd
            active_cmd_time = self.latest_active_cmd_time

        if image_msg is None or active_cmd is None:
            return
        if image_time is None or active_cmd_time is None:
            return
        if abs(image_time - active_cmd_time) > self.sync_tolerance:
            return

        if self.log_mode == 'corrections_only':
            if self.active_source != 'human':
                return

        source = self.active_source
        linear_vel = active_cmd.linear.x
        angular_vel = active_cmd.angular.z

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
