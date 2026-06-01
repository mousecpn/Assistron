from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped
import numpy as np
import time
# Define enum-like constants
LEFT_STICK_X = 0
LEFT_STICK_Y = 1
LEFT_TRIGGER = 2
RIGHT_STICK_X = 3
RIGHT_STICK_Y = 4
RIGHT_TRIGGER = 5
D_PAD_X = 6
D_PAD_Y = 7

A = 0
B = 1
X = 2
Y = 3
LEFT_BUMPER = 4
RIGHT_BUMPER = 5
CHANGE_VIEW = 6
MENU = 7
HOME = 8
LEFT_STICK_CLICK = 9
RIGHT_STICK_CLICK = 10

AXIS_DEFAULTS = {
    LEFT_TRIGGER: 1.0,
    RIGHT_TRIGGER: 1.0,
}


class JoyListener:
    def __init__(self):
        self.latest_axes = []
        self.latest_buttons = []

    def update_from_joy_msg(self, msg: Joy):
        self.latest_axes = list(msg.axes)
        self.latest_buttons = list(msg.buttons)
        # time.sleep(0.001)  # 短暂睡眠，确保数据更新后其他函数能读取到最新值

    def get_twist(self, stamp, frame_id="fr3_link0"):
        if not self.latest_axes or not self.latest_buttons:
            return None

        axes = self.latest_axes
        buttons = self.latest_buttons

        if len(axes) < 8 or len(buttons) < 11:
            print("JoyListener: Not enough axes or buttons received.")
            return None

        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = frame_id

        # Map axes to twist commands
        twist.twist.linear.x = -float(axes[RIGHT_STICK_X])
        twist.twist.linear.y = float(axes[RIGHT_STICK_Y])
        lin_z_trigger = 0.8  * (float(axes[RIGHT_TRIGGER]) - AXIS_DEFAULTS[RIGHT_TRIGGER])    # Down
        lin_z_bumper = 0.8 * float(buttons[RIGHT_BUMPER])   # Up
        twist.twist.linear.z = lin_z_trigger + lin_z_bumper
        
        twist.twist.angular.x = -float(axes[LEFT_STICK_X])
        twist.twist.angular.y = float(axes[LEFT_STICK_Y])
        ang_z_trigger = 0.8  * (float(axes[LEFT_TRIGGER]) - AXIS_DEFAULTS[LEFT_TRIGGER]) 
        ang_z_bumper = 0.8 * float(buttons[LEFT_BUMPER])
        twist.twist.angular.z = ang_z_trigger + ang_z_bumper

        return twist
    
    def get_twist_array(self):
        if not self.latest_axes or not self.latest_buttons:
            print("JoyListener: No joystick data received yet.")
            return None
        axes = self.latest_axes
        buttons = self.latest_buttons
        if len(axes) < 8 or len(buttons) < 11:
            print("JoyListener: Not enough axes or buttons received.")
            return None
        twist = np.zeros(11)  # [vx, vy, vz, wx, wy, wz, home, grasp, release, approach, drop]
        twist[0] = -float(axes[LEFT_STICK_Y])
        twist[1] = -float(axes[LEFT_STICK_X])
        lin_z_trigger = -0.8  if (float(axes[LEFT_TRIGGER]) - AXIS_DEFAULTS[LEFT_TRIGGER]) < -1.0 else 0.0   # Down
        lin_z_bumper = 0.8 * float(buttons[LEFT_BUMPER])   # Up
        twist[2] = lin_z_trigger + lin_z_bumper
        twist[3] = float(axes[RIGHT_STICK_X])
        twist[4] = -float(axes[RIGHT_STICK_Y])
        ang_z_trigger = -0.8  if (float(axes[RIGHT_TRIGGER]) - AXIS_DEFAULTS[RIGHT_TRIGGER]) < -1.0 else 0.0
        ang_z_bumper = 0.8 * float(buttons[RIGHT_BUMPER])
        twist[5] = ang_z_trigger + ang_z_bumper
        twist[6] = buttons[HOME]
        twist[7] = buttons[B]
        twist[8] = buttons[A]
        twist[9] = buttons[X]
        twist[10] = buttons[Y]
        twist[:3] *= 0.15
        twist[3:6] *= 0.4
        return twist
        

    def get_gripper_command(self):
        """Returns: 'close' / 'open' / None"""
        if not self.latest_buttons:
            return None

        if self.latest_buttons[A]:
            return 'close'
        elif self.latest_buttons[B]:
            return 'open'
        else:
            return None