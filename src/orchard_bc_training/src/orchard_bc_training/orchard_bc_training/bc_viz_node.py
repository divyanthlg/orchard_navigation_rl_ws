"""
BC Visualization Node — v0.7.2  (compressed image input)
=========================================================
Subscribes to the camera (CompressedImage), the policy command, the mux output
(actual cmd sent to the robot), and odom. Overlays two horizontal bars near the
bottom of the frame showing:

  Top bar   — ANGULAR velocity:  blue = policy cmd, white = odom measured
  Bottom bar — LINEAR velocity:   blue = policy cmd, white = odom measured

Republishes the annotated frame to /bc_viz/image (raw sensor_msgs/Image) for
viewing in rqt_image_view.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import cv2
import numpy as np
from cv_bridge import CvBridge


class BCVizNode(Node):
    def __init__(self):
        super().__init__('bc_viz')

        self.declare_parameter('image_topic',
            '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('odom_topic',
            '/w200_0100/platform/odom/filtered')
        self.declare_parameter('policy_cmd_topic', '/bc_policy/cmd_vel')
        self.declare_parameter('output_cmd_topic', '/w200_0100/cmd_vel')
        self.declare_parameter('viz_topic', '/bc_viz/image')
        self.declare_parameter('max_linear_vel',  1.0)   # bar full-scale
        self.declare_parameter('max_angular_vel', 0.5)

        image_topic     = self.get_parameter('image_topic').value
        odom_topic      = self.get_parameter('odom_topic').value
        policy_topic    = self.get_parameter('policy_cmd_topic').value
        output_topic    = self.get_parameter('output_cmd_topic').value
        viz_topic       = self.get_parameter('viz_topic').value
        self.max_lin    = self.get_parameter('max_linear_vel').value
        self.max_ang    = self.get_parameter('max_angular_vel').value

        self.bridge = CvBridge()
        self.policy_cmd = Twist()
        self.mux_cmd    = Twist()
        self.odom_lin   = 0.0
        self.odom_ang   = 0.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self.create_subscription(
            CompressedImage, image_topic, self._image_cb, sensor_qos)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, sensor_qos)
        self.create_subscription(Twist, policy_topic, self._policy_cb, 10)
        self.create_subscription(Twist, output_topic, self._mux_cb, 10)

        self.pub = self.create_publisher(Image, viz_topic, 10)

        self.get_logger().info(
            f'BC viz ready\n'
            f'  Image in:    {image_topic} (CompressedImage)\n'
            f'  Image out:   {viz_topic} (Image, annotated)\n'
            f'  View with:   ros2 run rqt_image_view rqt_image_view {viz_topic}')

    # ── callbacks ─────────────────────────────────────────────────────
    def _policy_cb(self, msg):  self.policy_cmd = msg
    def _mux_cb(self, msg):     self.mux_cmd = msg

    def _odom_cb(self, msg):
        self.odom_lin = float(msg.twist.twist.linear.x)
        self.odom_ang = float(msg.twist.twist.angular.z)

    def _image_cb(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
            if frame is None:
                raise RuntimeError('cv2.imdecode returned None')
        except Exception as e:
            self.get_logger().error(f'Image decode failed: {e}',
                                    throttle_duration_sec=2.0)
            return

        annotated = self._overlay(frame)

        try:
            out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out.header = msg.header
            self.pub.publish(out)
        except Exception as e:
            self.get_logger().error(f'Publish failed: {e}',
                                    throttle_duration_sec=2.0)

    # ── rendering ─────────────────────────────────────────────────────
    def _draw_bar(self, img, center_xy, w, h, value, max_abs, color_marker,
                  color_ref, ref_value, label):
        """
        Draw one horizontal bar centered at center_xy (cx, cy). Width w, height h.
        Black border, gray midline, white marker for `ref_value`, colored marker
        for `value`. Values outside [-max_abs, +max_abs] get clipped.
        """
        cx, cy = center_xy
        x1, x2 = cx - w // 2, cx + w // 2
        y1, y2 = cy - h // 2, cy + h // 2

        # Semi-transparent dark background
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        # Frame
        cv2.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), 2)
        # Zero line
        cv2.line(img, (cx, y1), (cx, y2), (160, 160, 160), 1)

        def marker(val, color, thickness):
            clipped = max(-max_abs, min(max_abs, val))
            frac = clipped / max_abs                       # -1..+1
            mx = int(cx + frac * (w // 2 - 2))
            cv2.line(img, (mx, y1 - 2), (mx, y2 + 2), color, thickness)

        marker(ref_value, color_ref, 2)     # white (odom)
        marker(value,     color_marker, 3)  # blue (policy)

        # Label above bar
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1,
                    cv2.LINE_AA)

    def _overlay(self, frame):
        h, w = frame.shape[:2]
        bar_w = int(w * 0.35)
        bar_h = 14
        cx = w // 2

        # Bottom of frame, two stacked bars
        ang_cy = h - 70
        lin_cy = h - 35

        BLUE  = (255, 120, 40)    # BGR: bright blue-ish
        WHITE = (255, 255, 255)

        # Angular
        self._draw_bar(
            frame, (cx, ang_cy), bar_w, bar_h,
            value=self.policy_cmd.angular.z,
            max_abs=self.max_ang,
            color_marker=BLUE, color_ref=WHITE,
            ref_value=self.odom_ang,
            label=f'ANG  policy={self.policy_cmd.angular.z:+.2f}  '
                  f'odom={self.odom_ang:+.2f}  (±{self.max_ang:.1f})',
        )

        # Linear
        self._draw_bar(
            frame, (cx, lin_cy), bar_w, bar_h,
            value=self.policy_cmd.linear.x,
            max_abs=self.max_lin,
            color_marker=BLUE, color_ref=WHITE,
            ref_value=self.odom_lin,
            label=f'LIN  policy={self.policy_cmd.linear.x:+.2f}  '
                  f'odom={self.odom_lin:+.2f}  (±{self.max_lin:.1f})',
        )

        # Source indicator: is the MUX currently forwarding policy or human?
        diff_lin = abs(self.mux_cmd.linear.x  - self.policy_cmd.linear.x)
        diff_ang = abs(self.mux_cmd.angular.z - self.policy_cmd.angular.z)
        source_is_policy = (diff_lin < 1e-3 and diff_ang < 1e-3)
        tag = '[POLICY]' if source_is_policy else '[HUMAN ]'
        color = (120, 255, 120) if source_is_policy else (80, 80, 255)
        cv2.putText(frame, tag, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        return frame


def main(args=None):
    rclpy.init(args=args)
    node = BCVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
