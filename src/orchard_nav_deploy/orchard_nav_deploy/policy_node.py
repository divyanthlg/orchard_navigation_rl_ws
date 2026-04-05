"""
Policy Node
=============
Loads the trained BC policy (VAE encoder + compression head + policy MLP)
and publishes velocity commands from camera images.

Publishes to /policy/cmd_vel (NOT directly to /cmd_vel).
The cmd_vel_mux node decides whether policy or human commands reach the robot.
"""

import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

import torch
import numpy as np
from cv_bridge import CvBridge
from torchvision import transforms


class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('model_project_path',
            '/home/divyanthlg/projects/orchard_navigation/vae_stabilityai')
        self.declare_parameter('checkpoint_path',
            '/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/checkpoints/best.pt')
        self.declare_parameter('vae_model_id', 'stabilityai/sd-vae-ft-mse')
        self.declare_parameter('latent_dim', 128)
        self.declare_parameter('action_dim', 2)
        self.declare_parameter('extra_dim', 0)
        self.declare_parameter('hidden_dim', 256)
        self.declare_parameter('image_size', 256)
        self.declare_parameter('max_linear_vel', 1.0)
        self.declare_parameter('max_angular_vel', 0.5)
        self.declare_parameter('inference_rate_hz', 10.0)

        # Read params
        image_topic = self.get_parameter('image_topic').value
        model_project_path = self.get_parameter('model_project_path').value
        checkpoint_path = self.get_parameter('checkpoint_path').value
        vae_model_id = self.get_parameter('vae_model_id').value
        latent_dim = self.get_parameter('latent_dim').value
        action_dim = self.get_parameter('action_dim').value
        extra_dim = self.get_parameter('extra_dim').value
        hidden_dim = self.get_parameter('hidden_dim').value
        self.image_size = self.get_parameter('image_size').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        inference_rate = self.get_parameter('inference_rate_hz').value

        # ── Load model ──────────────────────────────────────────────────
        # Add the project to Python path so we can import the models
        sys.path.insert(0, model_project_path)
        from models import OrchardNavModel

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Loading model on {self.device}...')

        self.model = OrchardNavModel(
            vae_model_id=vae_model_id,
            latent_dim=latent_dim,
            action_dim=action_dim,
            extra_dim=extra_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.model.load_trainable(checkpoint_path)
        self.model.eval()
        self.get_logger().info('Model loaded successfully')

        # ── Preprocessing ───────────────────────────────────────────────
        self.bridge = CvBridge()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # ── State ───────────────────────────────────────────────────────
        self.latest_image = None
        self.inference_count = 0

        # ── Subscribers ─────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_sub = self.create_subscription(
            Image, image_topic, self._image_cb, sensor_qos)

        # ── Publishers ──────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/policy/cmd_vel', 10)
        self.active_pub = self.create_publisher(Bool, '/policy/active', 10)

        # ── Inference timer ─────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / inference_rate, self._inference_tick)

        self.get_logger().info(f'Policy node ready — publishing to /policy/cmd_vel at {inference_rate} Hz')

    def _image_cb(self, msg: Image):
        self.latest_image = msg

    @torch.no_grad()
    def _inference_tick(self):
        if self.latest_image is None:
            return

        try:
            # Convert ROS image → tensor
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, 'rgb8')
            tensor = self.transform(cv_image).unsqueeze(0).to(self.device)

            # Run model
            action = self.model(tensor).squeeze(0).cpu().numpy()

            # Publish cmd_vel
            twist = Twist()
            twist.linear.x = float(action[0] * self.max_linear_vel)
            twist.angular.z = float(action[1] * self.max_angular_vel)
            self.cmd_pub.publish(twist)

            # Signal that policy is active
            active_msg = Bool()
            active_msg.data = True
            self.active_pub.publish(active_msg)

            self.inference_count += 1
            if self.inference_count % 100 == 0:
                self.get_logger().info(
                    f'Inference #{self.inference_count} — '
                    f'lin={twist.linear.x:+.3f} ang={twist.angular.z:+.3f}')

        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Publish zero velocity on shutdown
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.get_logger().info('Policy node stopped — zero velocity sent')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
