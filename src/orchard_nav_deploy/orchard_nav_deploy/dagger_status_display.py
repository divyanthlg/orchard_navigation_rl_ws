"""
DAgger Status Display
======================
Prints a live terminal display showing current control source,
velocity commands, and DAgger collection stats.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class DAggerStatusDisplay(Node):
    def __init__(self):
        super().__init__('dagger_status_display')

        self.active_source = 'waiting...'
        self.cmd_vel = Twist()
        self.policy_cmd = Twist()

        self.create_subscription(String, '/mux/active_source', self._source_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(Twist, '/policy/cmd_vel', self._policy_cb, 10)

        self.timer = self.create_timer(0.2, self._display)

    def _source_cb(self, msg):
        self.active_source = msg.data

    def _cmd_cb(self, msg):
        self.cmd_vel = msg

    def _policy_cb(self, msg):
        self.policy_cmd = msg

    def _display(self):
        source = self.active_source.upper()
        indicator = '🤖 POLICY' if self.active_source == 'policy' else '🧑 HUMAN '

        lin = self.cmd_vel.linear.x
        ang = self.cmd_vel.angular.z
        p_lin = self.policy_cmd.linear.x
        p_ang = self.policy_cmd.angular.z

        print(
            f'\r  {indicator} | '
            f'cmd: lin={lin:+.3f} ang={ang:+.3f} | '
            f'policy: lin={p_lin:+.3f} ang={p_ang:+.3f}  ',
            end='', flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = DAggerStatusDisplay()
    print('\n  Orchard Nav — Live Status\n  ─────────────────────────')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
