"""
BC Cmd Vel Mux — v0.7  (self-contained inside orchard_bc_training)
====================================================================
Merges /bc_policy/cmd_vel with human RC teleop. Published result goes
to the robot's cmd_vel topic. Independent of orchard_nav_deploy.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class BCCmdVelMuxNode(Node):
    def __init__(self):
        super().__init__('bc_cmd_vel_mux')

        self.declare_parameter('policy_cmd_topic', '/bc_policy/cmd_vel')
        self.declare_parameter('human_cmd_topic', '/rc_teleop/cmd_vel')
        self.declare_parameter('output_cmd_topic', '/cmd_vel')
        self.declare_parameter('human_timeout_sec', 0.5)
        self.declare_parameter('joy_linear_deadzone', 0.05)
        self.declare_parameter('joy_angular_deadzone', 0.05)

        policy_topic = self.get_parameter('policy_cmd_topic').value
        human_topic = self.get_parameter('human_cmd_topic').value
        output_topic = self.get_parameter('output_cmd_topic').value
        self.human_timeout = self.get_parameter('human_timeout_sec').value
        self.joy_linear_dz = self.get_parameter('joy_linear_deadzone').value
        self.joy_angular_dz = self.get_parameter('joy_angular_deadzone').value

        self.active_source = 'policy'
        self.last_human_time = 0.0
        self.latest_policy_cmd = Twist()
        self.latest_human_cmd = Twist()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self.policy_sub = self.create_subscription(
            Twist, policy_topic, self._policy_cb, 10)
        self.human_sub = self.create_subscription(
            Twist, human_topic, self._human_cb, sensor_qos)

        self.output_pub = self.create_publisher(Twist, output_topic, 10)
        self.source_pub = self.create_publisher(String, '/bc_mux/active_source', 10)

        self.timer = self.create_timer(0.02, self._mux_tick)

        self.get_logger().info(
            f'BC cmd vel mux ready\n'
            f'  Policy input:  {policy_topic}\n'
            f'  Human input:   {human_topic}\n'
            f'  Output:        {output_topic}\n'
            f'  Human timeout: {self.human_timeout}s')

    def _policy_cb(self, msg):
        self.latest_policy_cmd = msg

    def _human_cb(self, msg):
        self.latest_human_cmd = msg
        lin = abs(msg.linear.x)
        ang = abs(msg.angular.z)
        if lin > self.joy_linear_dz or ang > self.joy_angular_dz:
            now = self.get_clock().now().nanoseconds / 1e9
            if self.active_source != 'human':
                self.get_logger().info('>>> HUMAN OVERRIDE <<<')
            self.active_source = 'human'
            self.last_human_time = now

    def _mux_tick(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.active_source == 'human':
            if (now - self.last_human_time) > self.human_timeout:
                self.get_logger().info('>>> Human released — back to policy <<<')
                self.active_source = 'policy'

        if self.active_source == 'human':
            self.output_pub.publish(self.latest_human_cmd)
        else:
            self.output_pub.publish(self.latest_policy_cmd)

        source_msg = String()
        source_msg.data = self.active_source
        self.source_pub.publish(source_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BCCmdVelMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.output_pub.publish(Twist())
            node.get_logger().info('Mux stopped — zero velocity sent')
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
