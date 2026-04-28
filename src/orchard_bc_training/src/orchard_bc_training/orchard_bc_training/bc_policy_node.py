"""
BC Policy Node — v0.7.2  (compressed image input)
==================================================
Sequence VAE+GRU+MLP policy. Distinct from the v0.6 policy_node in
orchard_nav_deploy: different node name, different output topic.

- Node name:     bc_policy_node
- Output topic:  /bc_policy/cmd_vel
- Active flag:   /bc_policy/active
- Image input:   sensor_msgs/CompressedImage (JPEG/PNG payload)

Maintains a seq_len=13 ring buffer of VAE latents and per-timestep extras.
VAE encode at 2 Hz (matches training), GRU+MLP publish at 10 Hz.
"""

import collections
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

import cv2
import torch
import numpy as np
from torchvision import transforms

from orchard_bc_training.models import OrchardNavModel
from orchard_bc_training.vae_resolve import resolve_vae_id


DEFAULT_CHECKPOINT = os.path.expanduser(
    '~/ros2/orchard_navigation_rl_ws/checkpoints/best.pt')


class BCPolicyNode(Node):
    def __init__(self):
        super().__init__('bc_policy_node')

        self.declare_parameter('image_topic',
            '/sensors/camera_0/color/compressed')
        self.declare_parameter('odom_topic',
            '/platform/odom/filtered')
        self.declare_parameter('checkpoint_path', DEFAULT_CHECKPOINT)
        self.declare_parameter('vae_model_id', '')  # empty → auto-resolve
        self.declare_parameter('seq_len', 13)
        self.declare_parameter('image_size', 256)
        self.declare_parameter('max_linear_vel', 1.0)
        self.declare_parameter('max_angular_vel', 0.5)
        self.declare_parameter('perception_rate_hz', 2.0)
        self.declare_parameter('command_rate_hz', 10.0)
        self.declare_parameter('output_cmd_topic', '/bc_policy/cmd_vel')

        image_topic     = self.get_parameter('image_topic').value
        odom_topic      = self.get_parameter('odom_topic').value
        checkpoint_path = os.path.expanduser(
            self.get_parameter('checkpoint_path').value)
        vae_model_id    = resolve_vae_id(self.get_parameter('vae_model_id').value)
        self.get_logger().info(f'VAE source: {vae_model_id}')
        self.seq_len    = self.get_parameter('seq_len').value
        self.image_size = self.get_parameter('image_size').value
        self.max_lin    = self.get_parameter('max_linear_vel').value
        self.max_ang    = self.get_parameter('max_angular_vel').value
        perception_rate = self.get_parameter('perception_rate_hz').value
        command_rate    = self.get_parameter('command_rate_hz').value
        output_topic    = self.get_parameter('output_cmd_topic').value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Loading model on {self.device}...')
        self.model = OrchardNavModel(
            vae_model_id=vae_model_id,
            seq_len=self.seq_len,
            load_vae=True,
        ).to(self.device)
        self.model.load_trainable(checkpoint_path, map_location=self.device)
        self.model.eval()
        self.get_logger().info(f'Model loaded from {checkpoint_path}')

        # Transform: expects a numpy HxWx3 uint8 RGB image (ToPILImage handles it)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.latent_buf = collections.deque(maxlen=self.seq_len)
        self.extra_buf  = collections.deque(maxlen=self.seq_len)

        self.latest_image = None
        self.cur_lin_odom = 0.0
        self.cur_ang_odom = 0.0
        self.last_pub_lin = 0.0
        self.last_pub_ang = 0.0
        self.inference_count = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.image_sub = self.create_subscription(
            CompressedImage, image_topic, self._image_cb, sensor_qos)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_cb, sensor_qos)

        self.cmd_pub = self.create_publisher(Twist, output_topic, 10)
        self.active_pub = self.create_publisher(Bool, '/bc_policy/active', 10)

        self.perception_timer = self.create_timer(
            1.0 / perception_rate, self._perception_tick)
        self.command_timer = self.create_timer(
            1.0 / command_rate, self._command_tick)

        self.get_logger().info(
            f'BC policy ready\n'
            f'  Image topic: {image_topic} (CompressedImage)\n'
            f'  Perception:  {perception_rate} Hz (VAE encode → ring buffer)\n'
            f'  Command:     {command_rate} Hz → {output_topic}\n'
            f'  seq_len:     {self.seq_len} '
            f'({self.seq_len / perception_rate:.1f} s history)\n'
            f'  Waiting {self.seq_len / perception_rate:.1f}s '
            f'before first real publish.')

    def _image_cb(self, msg: CompressedImage):
        self.latest_image = msg

    def _odom_cb(self, msg: Odometry):
        self.cur_lin_odom = float(msg.twist.twist.linear.x)
        self.cur_ang_odom = float(msg.twist.twist.angular.z)

    @staticmethod
    def _decode_compressed_rgb(msg: CompressedImage):
        """Decode CompressedImage payload into an HxWx3 RGB uint8 ndarray."""
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def _perception_tick(self):
        if self.latest_image is None:
            self.get_logger().warn('No image yet', throttle_duration_sec=3.0)
            return
        try:
            rgb = self._decode_compressed_rgb(self.latest_image)
            if rgb is None:
                raise RuntimeError('cv2.imdecode returned None')
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            latent = self.model.encode_images(tensor).squeeze(0)
        except Exception as e:
            self.get_logger().error(
                f'VAE encode failed: {e}', throttle_duration_sec=2.0)
            return

        extra = torch.tensor(
            [self.cur_lin_odom, self.cur_ang_odom,
             self.last_pub_lin, self.last_pub_ang],
            dtype=torch.float32, device=self.device)

        self.latent_buf.append(latent)
        self.extra_buf.append(extra)

    @torch.no_grad()
    def _command_tick(self):
        if len(self.latent_buf) < self.seq_len:
            self.cmd_pub.publish(Twist())
            return

        try:
            lat = torch.stack(list(self.latent_buf), dim=0).unsqueeze(0)
            ext = torch.stack(list(self.extra_buf), dim=0).unsqueeze(0)
            actions = self.model(lat, ext, is_latents=True)
            a = actions[0, -1].cpu().numpy()

            lin_cmd = float(np.clip(a[0], -1.0, 1.0)) * self.max_lin
            ang_cmd = float(np.clip(a[1], -1.0, 1.0)) * self.max_ang

            twist = Twist()
            twist.linear.x = lin_cmd
            twist.angular.z = ang_cmd
            self.cmd_pub.publish(twist)

            self.last_pub_lin = lin_cmd
            self.last_pub_ang = ang_cmd

            active = Bool(); active.data = True
            self.active_pub.publish(active)

            self.inference_count += 1
            if self.inference_count % 100 == 0:
                self.get_logger().info(
                    f'Inference #{self.inference_count} — '
                    f'lin={lin_cmd:+.3f} ang={ang_cmd:+.3f}')
        except Exception as e:
            self.get_logger().error(
                f'Inference failed: {e}', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = BCPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
            node.get_logger().info('BC policy stopped — zero velocity sent')
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
