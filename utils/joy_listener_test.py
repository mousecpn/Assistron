#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped

# 导入你写好的类 (假设原始代码保存在 joy_listener.py 中)
from utils.joy_listener import JoyListener

class JoyTesterNode(Node):
    def __init__(self):
        super().__init__('real_joy_tester')
        
        # 实例化你的 Listener
        self.listener = JoyListener()
        
        # 订阅手柄基础驱动发出的 /joy 话题，队列长度设为 10
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        
        self.get_logger().info("正在监听真实的 /joy 话题...")
        self.get_logger().info("请随意拨动摇杆或按下按键进行测试 (按 Ctrl+C 退出)")

    def joy_callback(self, msg):
        """
        ROS 2 订阅器回调函数，每次手柄状态更新时触发
        """
        # 1. 更新手柄数据
        self.listener.update_from_joy_msg(msg)

        # 2. 打印分割线，方便终端查看
        print("\n" + "="*30)
        print(" 🎮 手柄输入已接收")
        print("="*30)
        
        # 3. 测试 get_twist (ROS Message 格式)
        # 获取当前 ROS 2 时间戳
        current_time = self.get_clock().now().to_msg()
        twist_msg = self.listener.get_twist(stamp=current_time, frame_id="test_frame")
        
        if twist_msg:
            print("[ROS Twist 输出]")
            print(f"  Linear  (x, y, z): ({twist_msg.twist.linear.x: .2f}, {twist_msg.twist.linear.y: .2f}, {twist_msg.twist.linear.z: .2f})")
            print(f"  Angular (x, y, z): ({twist_msg.twist.angular.x: .2f}, {twist_msg.twist.angular.y: .2f}, {twist_msg.twist.angular.z: .2f})")
        else:
            print("[ROS Twist 输出] 数据不足")

        # 4. 测试 get_twist_array (Numpy Array 格式)
        twist_arr = self.listener.get_twist_array()
        if twist_arr is not None:
            print("\n[Numpy Array 输出]")
            # 格式化数组输出，保留两位小数
            arr_str = ", ".join([f"{x: .2f}" for x in twist_arr])
            print(f"  [{arr_str}]")
        else:
            print("\n[Numpy Array 输出] 数据不足")

        # 5. 测试夹爪开合指令
        gripper_cmd = self.listener.get_gripper_command()
        print("\n[夹爪指令输出]")
        if gripper_cmd:
            print(f"  动作: {gripper_cmd.upper()}")
        else:
            print("  动作: 无 (None)")

def main(args=None):
    # 初始化 rclpy
    rclpy.init(args=args)
    
    # 创建节点实例
    tester_node = JoyTesterNode()
    
    try:
        # 保持节点运行
        rclpy.spin(tester_node)
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 退出事件
        pass
    finally:
        # 优雅地销毁节点和关闭 rclpy
        tester_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()