#!/usr/bin/env python3
"""
FR3 Robot Controller using ROS2 and MoveIt (ZMQ Client Version)
"""

import rclpy
import sys
import zmq
import pickle

import sensor_msgs
from utils.transform import Transform, reorder_pose_list, Rotation, matrix_to_euler_angles
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState

import time
from typing import List, Optional
from threading import Lock

from franka_msgs.action import ErrorRecovery
from franka_msgs.msg import FrankaRobotState, Errors
from franka_msgs.srv import SetForceTorqueCollisionBehavior
from control_msgs.action import GripperCommand

from rclpy.callback_groups import ReentrantCallbackGroup
# from utils_exp import vis # 可选
import numpy as np

# 注意：Client 端不需要导入 curobo 的库，除非你需要 Mesh 类的辅助函数来生成 dict
# 这里我们假设 Mesh 数据处理逻辑稍作修改，或者保留 Mesh 引用仅用于数据结构转换
from curobo.geom.types import Mesh, WorldConfig 

from utils.control import calculate_velocity
import roboticstoolbox as rtb
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

def broadcast_static(transform: 'Transform', parent_frame: str, child_frame: str):
    """
    Broadcast a one-shot static TF frame via a temporary rclpy node.
    transform : Transform  (rotation as scipy Rotation, translation as [x,y,z])
    parent_frame : str     e.g. 'fr3_hand_tcp'
    child_frame  : str     e.g. 'robotiq_tcp'
    """
    node = rclpy.create_node('_static_tf_broadcaster_tmp')
    broadcaster = tf2_ros.StaticTransformBroadcaster(node)

    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame

    tr = transform.translation
    t.transform.translation.x = float(tr[0])
    t.transform.translation.y = float(tr[1])
    t.transform.translation.z = float(tr[2])

    q = transform.rotation.as_quat()  # [x, y, z, w]
    t.transform.rotation.x = float(q[0])
    t.transform.rotation.y = float(q[1])
    t.transform.rotation.z = float(q[2])
    t.transform.rotation.w = float(q[3])

    broadcaster.sendTransform(t)
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()

# --- 新增：ZMQ 客户端类 ---
class MotionPlannerClient:
    def __init__(self, host='localhost', port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"[MotionClient] Connected to Motion Server at {port}")
        
        # 简单的 Ping 测试
        try:
            self.socket.send(pickle.dumps({'cmd': 'ping'}))
            # 设置超时防止卡死
            if self.socket.poll(2000): # 2s timeout
                self.socket.recv()
                print("[MotionClient] Server online.")
            else:
                print("[MotionClient] WARNING: Server not responding!")
        except Exception as e:
            print(f"[MotionClient] Init error: {e}")

    def update_world(self, cuboids=None, pcl=None):
        """
        发送环境信息给 Server
        :param cuboids: dict, e.g. {'table': {'dims':..., 'pose':...}}
        :param pcl: numpy array (N, 3), 原始点云数据
        """
        # 预处理：如果 cuboids 嵌套在 'cuboid' 键下，提取出来，因为 Server 端会重新组装
        cuboids_data = cuboids
        if cuboids is not None and 'cuboid' in cuboids:
            cuboids_data = cuboids['cuboid']

        req = {
            'cmd': 'update_world',
            'cuboids': cuboids_data,
            'pcl': pcl # 直接发送 Numpy 数组，pickle 会自动处理
        }
        
        try:
            self.socket.send(pickle.dumps(req))
            resp = pickle.loads(self.socket.recv())
            return resp.get('success', False)
        except Exception as e:
            print(f"[MotionClient] Update World Error: {e}")
            self._reset_socket()
            return False

    def plan(self, start_joint_state, target_pose, plan_config=None):
        """
        返回: (waypoints_np, success)
        """
        req = {
            'cmd': 'plan',
            'start': start_joint_state,
            'target': target_pose,
            # plan_config 暂不支持传输复杂对象，Server 端使用默认
        }
        
        try:
            self.socket.send(pickle.dumps(req))
            resp = pickle.loads(self.socket.recv())
            
            success = resp.get('success', False)
            waypoints = resp.get('waypoints', None)
            
            return waypoints, success
            
        except Exception as e:
            print(f"[MotionClient] Plan Error: {e}")
            self._reset_socket()
            return None, False

    def _reset_socket(self):
        print("[MotionClient] Resetting socket...")
        self.socket.close()
        self.context.term()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5556")


class PandaCommander(Node):
    """
    FR3 Robot Controller with ZMQ Motion Planner
    """
    
    def __init__(self, robot_name: str = "fr3"):
        super().__init__('fr3_controller')
        self.callback_group = ReentrantCallbackGroup()
        
        # Robot parameters
        self.robot_name = robot_name
        self.joint_names = [
            f'{robot_name}_joint1', f'{robot_name}_joint2', f'{robot_name}_joint3', 
            f'{robot_name}_joint4', f'{robot_name}_joint5', f'{robot_name}_joint6', 
            f'{robot_name}_joint7'
        ]
        
        # self.T_body_tcp = Transform.from_dict({"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [0.0, 0.0, -0.05]})
        # self.T_tcp_body = self.T_body_tcp.inverse()
        # self.T_tcp_link8 = self.T_body_tcp # Alias

        self.T_link8_tcp = Transform.from_dict({"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [0.0, 0.0, 0.05]})
        self.T_tcp_link8 = self.T_link8_tcp.inverse()


        # wrench 
        self.wrench_raw = np.zeros(6)
        self.wrench_baseline = np.zeros(6)
        self.wrench_filtered = np.zeros(6)
        self.baseline_samples = []
        self.baseline_ready = False
        self.baseline_sample_count = 100        # 前100帧用于基线估计
        self.ema_alpha = 0.02                   # 低通滤波系数（慢）
        self.contact_hysteresis_count = 3       # 连续帧数确认接触
        self._contact_count = 0
        
        # --- ROS Subs/Pubs ---
        # self.control_topic = "/joint_velocity_controller/joint_velocity"
        self.control_topic = '/joint_position_impedance_controller/joint_states'
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1, callback_group=self.callback_group)
        self.joint_velo_pub = self.create_publisher(JointState, self.control_topic, 1, callback_group=self.callback_group)
        # self.joint_velo_pub = self.create_publisher(JointState, '/joint_position_impedance_controller/joint_states', 1, callback_group=self.callback_group)

        self.predicted_path_pub = self.create_publisher(Path, '/planned_trajectory', 1, callback_group=self.callback_group)
        self.last_path_pub = self.create_publisher(Path, '/last_trajectory', 1, callback_group=self.callback_group)
        self.action_samples_pub = self.create_publisher(MarkerArray, '/action_samples_viz', 1, callback_group=self.callback_group)

        self.force_sub = self.create_subscription(FrankaRobotState, "/franka_robot_state_broadcaster/robot_state", self.force_callback, 10, callback_group=self.callback_group)
        self.robot_state_sub = self.create_subscription(FrankaRobotState, 'franka_robot_state_broadcaster/robot_state', self.robot_state_cb, 10, callback_group=self.callback_group)

        # --- Clients ---
        # Robotiq gripper client
        self.gripper_client = ActionClient(
            self, 
            GripperCommand, 
            '/robotiq/robotiq_gripper_controller/gripper_cmd', 
            callback_group=self.callback_group
        )
        self.error_recovery_client = ActionClient(self, ErrorRecovery, '/action_server/error_recovery', callback_group=self.callback_group)
        self.collision_client = self.create_client(SetForceTorqueCollisionBehavior, '/service_server/set_force_torque_collision_behavior')
        
        # --- 初始化 Motion Planner Client (替代原有的 MotionPlanner) ---
        self.planner = MotionPlannerClient() 
        
        # PID & Limits
        self.kp = 1.0
        self.kd = 0.04
        self.dt = 0.03
        self.limit_vel = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]) * 0.2
        self.limit_acc = np.array([15, 7.5, 10, 12.5, 15, 20, 20]) * 0.04
        self.goal_tolerance = 0.01
        
        self.joint_command_msg = JointState()
        self.react_control_flag = False
        self._trajectory_lock = Lock()
        self._trajectory_queue: List[np.ndarray] = []
        self._trajectory_last_target: Optional[np.ndarray] = None
        self._trajectory_alignment = True
        self._trajectory_alignment_start: Optional[float] = None
        self._trajectory_alignment_timeout = 3.0
        self._trajectory_process_tolerance = 0.4
        self.force_limit = 10.0
        self.wrench = np.zeros(6)
        self.robot_error = False
        # self.home_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        # self.home_joints = [-0.5049881727597879, -0.8549570584577496, 0.31917120002324684, -2.358842981033868, 0.22935616762328961, 1.5388877849995934, 1.5388877849995934]  # 0.5157969213680381
        # self.home_joints = [1.57/2, -0.6283,  0.0000, -2.5133,  0.0000,  1.8850,  0.0000]
        self.home_joints = [0.0, -0.6283,  0.0000, -2.5133,  0.0000,  1.8850,  0.0000]
        self.current_joint_state = None
        
        # Kinematics Model (for Client side FK/IK utils)
        self.panda = rtb.models.Panda()

        # Init World Config (Table)
        self.table_cfg = None

        # Kalman filter for joint position estimation
        self._init_joint_kalman()

        self.create_timer(0.01, self.joint_velo_publisher, callback_group=self.callback_group)
        if 'velocity' not in self.control_topic:
            self.create_timer(self.dt, self._trajectory_queue_worker, callback_group=self.callback_group)
        
        # Wait for robot state
        self._wait_for_connection()
        self.set_high_collision_thresholds()

        rot_matrix = Rotation.from_euler('z', np.pi/4).as_matrix() # Try +np.pi/4 if fingers are 90 deg off
        self.T_tcp_robotiq = Transform(Rotation.from_matrix(rot_matrix), [0.0, 0.0, 0.040]) # robotiq
        self.T_robotiq_tcp = self.T_tcp_robotiq.inverse()


        broadcast_static(
            self.T_tcp_robotiq, "fr3_hand_tcp", "robotiq_tcp"
        )


    def _wait_for_connection(self):
        timeout = 10.0
        start_time = time.time()
        while self.current_joint_state is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.current_joint_state:
            self.get_logger().info("Controller connected to Robot.")
        else:
            self.get_logger().error("Robot Joint State Timeout.")

    def set_high_collision_thresholds(self):
        """把碰撞阈值调高一点，用于允许抓取/接触环境"""
        
        # 等待服务
        if not self.collision_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Collision behavior service not available.")
            return

        request = SetForceTorqueCollisionBehavior.Request()

        # ========= 设置较高的阈值（常用配置） =========
        # request.lower_torque_thresholds_acc  = [30, 30, 30, 20, 20, 15, 10]
        # request.upper_torque_thresholds_acc  = [45, 45, 45, 35, 35, 25, 20]

        # request.lower_force_thresholds_acc   = [50, 50, 50, 30, 30, 30]
        # request.upper_force_thresholds_acc   = [80, 80, 80, 50, 50, 50]

        request.lower_torque_thresholds_nominal = [20., 20., 20., 15., 15., 10., 10.]
        request.upper_torque_thresholds_nominal = [35., 35., 35., 25., 25., 20., 15.]

        request.lower_force_thresholds_nominal  = [40., 40., 40., 25., 25., 25.]
        request.upper_force_thresholds_nominal  = [100., 100., 100., 100., 100., 100.]
        # ========================================

        future = self.collision_client.call_async(request)

        # 回调处理结果
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info("High collision thresholds successfully set!")
        else:
            self.get_logger().error("Failed to set collision thresholds.")
    
    def visualize_trajectory(self, joint_waypoints, panda):
        """Visualize planned trajectory as a Path message"""
        trajectory = []
        for joints in joint_waypoints:
            T_ee = panda.fkine(joints)
            trajectory.append(np.array(T_ee))
        path_msg = self.create_path_msg(trajectory, frame_id="fr3_link0")
        self.predicted_path_pub.publish(path_msg)
    
    def visualize_last_trajectory(self, joint_waypoints, panda):
        """Visualize planned trajectory as a Path message"""
        trajectory = []
        for joints in joint_waypoints:
            T_ee = panda.fkine(joints)
            trajectory.append(np.array(T_ee))
        path_msg = self.create_path_msg(trajectory, frame_id="fr3_link0")
        self.last_path_pub.publish(path_msg)

    def visualize_action_samples(self, action_samples, cur_q, panda, dt=1.0/15.0):
        """
        Visualize N sampled action chunks as LINE_STRIP markers in RViz.
        Each sample is integrated from cur_q using joint velocities and FK.

        action_samples : np.ndarray (N, H, D)  — joint velocity chunks (first 7 dims used)
        cur_q          : np.ndarray (7,)        — current joint configuration
        panda          : roboticstoolbox robot
        dt             : float                  — controller timestep
        """
        N, H, _ = action_samples.shape
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for i in range(N):
            marker = Marker()
            marker.header.frame_id = "fr3_link0"
            marker.header.stamp = stamp
            marker.ns = "action_samples"
            marker.id = i + 1  # reserve id=0 for the DELETEALL marker
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.003  # line width in metres
            marker.pose.orientation.w = 1.0
            # Color ramp: blue → red across samples
            t = i / max(N - 1, 1)
            marker.color = ColorRGBA(r=float(t), g=0.3, b=float(1.0 - t), a=0.6)

            q = np.array(cur_q, dtype=np.float64)
            for step in range(H):
                T_mat = np.array(panda.fkine(q))
                pt = Point(x=float(T_mat[0, 3]), y=float(T_mat[1, 3]), z=float(T_mat[2, 3]))
                marker.points.append(pt)
                q = q + action_samples[i, step, :7] * dt

            marker_array.markers.append(marker)

        # Prepend DELETEALL to clear stale markers from previous call
        del_marker = Marker()
        del_marker.header.frame_id = "fr3_link0"
        del_marker.header.stamp = stamp
        del_marker.ns = "action_samples"
        del_marker.action = Marker.DELETEALL
        marker_array.markers.insert(0, del_marker)

        self.action_samples_pub.publish(marker_array)

    def create_path_msg(self, trajectory, frame_id="world", ee_frame_id="cur_link", color_alpha=1.0):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id
        for pose_matrix in trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(pose_matrix[0, 3])
            pose_stamped.pose.position.y = float(pose_matrix[1, 3])
            pose_stamped.pose.position.z = float(pose_matrix[2, 3])
            rot_matrix = pose_matrix[:3, :3]
            quat = Rotation.from_matrix(rot_matrix).as_quat()
            pose_stamped.pose.orientation.x = float(quat[0])
            pose_stamped.pose.orientation.y = float(quat[1])
            pose_stamped.pose.orientation.z = float(quat[2])
            pose_stamped.pose.orientation.w = float(quat[3])
            path_msg.poses.append(pose_stamped)
        return path_msg

    def joint_velo_publisher(self):
        try:
            if self.react_control_flag:
                self.joint_command_msg.header.stamp = self.get_clock().now().to_msg()
                self.joint_velo_pub.publish(self.joint_command_msg)
        except Exception as e:
            pass

    def set_joint_trajectory_queue(self, joint_waypoints: List[List[float]], replace: bool = True, alignment: bool = True) -> bool:
        waypoints = np.asarray(joint_waypoints, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[1] != 7 or len(waypoints) == 0:
            return False

        with self._trajectory_lock:
            if replace:
                self._trajectory_queue = [wp.copy() for wp in waypoints]
            else:
                self._trajectory_queue.extend([wp.copy() for wp in waypoints])
            self._trajectory_last_target = waypoints[-1].copy()
            self._trajectory_alignment = alignment
            self._trajectory_alignment_start = None

        self.joint_command_msg.name = self.joint_names
        self.react_control_flag = True
        return True

    def clear_joint_trajectory_queue(self):
        with self._trajectory_lock:
            self._trajectory_queue = []
            self._trajectory_last_target = None
            self._trajectory_alignment_start = None
        self.react_control_flag = False

    def is_trajectory_running(self) -> bool:
        with self._trajectory_lock:
            return len(self._trajectory_queue) > 0

    def _trajectory_queue_worker(self):
        if not self.react_control_flag:
            return

        current_joint = self.get_current_joint_position()
        if current_joint is None:
            return
        current_joint = np.array(current_joint, dtype=np.float64)

        with self._trajectory_lock:
            if len(self._trajectory_queue) == 0:
                self.react_control_flag = False
                self.joint_command_msg.name = self.joint_names
                self.joint_command_msg.position = current_joint.tolist()
                return
            target = self._trajectory_queue[0]
            final_target = None if self._trajectory_last_target is None else self._trajectory_last_target.copy()
            alignment = self._trajectory_alignment

        error = np.linalg.norm(target - current_joint)
        if error < self._trajectory_process_tolerance:
            with self._trajectory_lock:
                if len(self._trajectory_queue) > 0:
                    self._trajectory_queue.pop(0)
                    # print(f"Trajectory queue: popped reached target, {len(self._trajectory_queue)} waypoints left.")
                queue_empty = len(self._trajectory_queue) == 0
                if not queue_empty:
                    target = self._trajectory_queue[0]

            if queue_empty:
                if alignment and final_target is not None:
                    final_error = np.linalg.norm(final_target - current_joint)
                    if final_error > self.goal_tolerance:
                        with self._trajectory_lock:
                            if self._trajectory_alignment_start is None:
                                self._trajectory_alignment_start = time.time()
                            align_elapsed = time.time() - self._trajectory_alignment_start
                        if align_elapsed < self._trajectory_alignment_timeout:
                            self.joint_command_msg.name = self.joint_names
                            self.joint_command_msg.position = final_target.tolist()
                            return
                self.react_control_flag = False
                self.joint_command_msg.name = self.joint_names
                self.joint_command_msg.position = current_joint.tolist()
                return

        self.joint_command_msg.name = self.joint_names
        self.joint_command_msg.position = target.tolist()

    
    def force_callback(self, msg):
        """接收传感器回调后：保存原始、并做一个简单低通滤波（EMA）到 wrench_filtered"""
        raw = np.array([
            msg.o_f_ext_hat_k.wrench.force.x,
            msg.o_f_ext_hat_k.wrench.force.y,
            msg.o_f_ext_hat_k.wrench.force.z,
            msg.o_f_ext_hat_k.wrench.torque.x,
            msg.o_f_ext_hat_k.wrench.torque.y,
            msg.o_f_ext_hat_k.wrench.torque.z,
        ])
        self.wrench_raw = raw

        # 启动时用前 baseline_sample_count 帧做平均基线
        if not getattr(self, 'baseline_ready', False):
            self.baseline_samples.append(raw)
            if len(self.baseline_samples) >= self.baseline_sample_count:
                self.wrench_baseline = np.mean(self.baseline_samples, axis=0)
                self.baseline_ready = True
        else:
            # 运行时用 EMA 慢速更新 baseline（可选）
            self.wrench_baseline = (1 - self.ema_alpha) * self.wrench_baseline + self.ema_alpha * raw

        # 对原始信号做一个 EMA 低通以减少高频噪声
        self.wrench_filtered = (1 - self.ema_alpha) * getattr(self, 'wrench_filtered', np.zeros(6)) + self.ema_alpha * raw


    def robot_state_cb(self, msg):
        detected_error = False
        self.robot_mode = msg.robot_mode
        if np.any([msg.collision_indicators.is_cartesian_angular_collision.x, msg.collision_indicators.is_cartesian_angular_collision.y, msg.collision_indicators.is_cartesian_angular_collision.z]):
            detected_error = True
        for s in Errors.__slots__:
            if getattr(msg.current_errors, s):
                detected_error = True
        if not self.robot_error and detected_error:
            self.robot_error = True
            self.get_logger().warn("Detected robot error")
        if self.robot_error and not detected_error:
            self.robot_error = False
            self.get_logger().info("Robot error cleared")

    def joint_state_callback(self, msg: JointState):
        self.current_joint_state = msg
        # Feed raw positions into Kalman filter
        positions = []
        for joint_name in self.joint_names:
            try:
                idx = msg.name.index(joint_name)
                positions.append(msg.position[idx])
            except ValueError:
                return
        self._update_kalman(np.array(positions, dtype=np.float64))

    def _init_joint_kalman(self) -> None:
        """Initialise a constant-velocity Kalman filter for 7-DOF joint positions.

        State  x = [q (7), dq (7)]  (14-D)
        Obs    z = q                 (7-D)
        """
        n = 7
        dt = self.dt  # nominal control period

        # State-transition: position integrates velocity
        F = np.eye(2 * n)
        F[:n, n:] = np.eye(n) * dt
        self._kf_F = F

        # Observation matrix: measure positions only
        H = np.zeros((n, 2 * n))
        H[:n, :n] = np.eye(n)
        self._kf_H = H

        # Process noise (tunable)
        self._kf_Q = np.diag([1e-4] * n + [1e-2] * n)

        # Measurement noise (encoder resolution ~0.1 mrad)
        self._kf_R = np.eye(n) * 1e-5

        # State & covariance
        self._kf_x = np.zeros(2 * n)
        self._kf_P = np.eye(2 * n)
        self._kf_initialized = False

    def _update_kalman(self, z: np.ndarray) -> None:
        """Predict + update step given a new 7-D joint position measurement."""
        n = 7
        if not self._kf_initialized:
            self._kf_x[:n] = z
            self._kf_initialized = True
            return

        F, H = self._kf_F, self._kf_H
        Q, R = self._kf_Q, self._kf_R

        # Predict
        x_pred = F @ self._kf_x
        P_pred = F @ self._kf_P @ F.T + Q

        # Update (Joseph form for numerical stability)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.solve(S.T, np.eye(n)).T  # K = P H^T S^{-1}
        I_KH = np.eye(2 * n) - K @ H
        self._kf_x = x_pred + K @ (z - H @ x_pred)
        self._kf_P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

    def get_estimated_joint_position(self) -> Optional[List[float]]:
        """Return the Kalman-filtered joint position estimate (7 values).

        Falls back to the raw encoder reading until the filter is initialised.
        """
        if not getattr(self, '_kf_initialized', False):
            return self.get_current_joint_position()
        return self._kf_x[:7].tolist()

    def get_estimated_joint_velocity(self) -> Optional[List[float]]:
        """Return the Kalman-estimated joint velocity (7 values)."""
        if not getattr(self, '_kf_initialized', False):
            return None
        return self._kf_x[7:14].tolist()

    def get_current_joint_position(self) -> Optional[List[float]]:
        if self.current_joint_state is None:
            return None
        joint_positions = []
        for joint_name in self.joint_names:
            try:
                idx = self.current_joint_state.name.index(joint_name)
                joint_positions.append(self.current_joint_state.position[idx])
            except ValueError:
                return None
        return joint_positions

    def get_ee_pose(self):
        joint_positions = self.get_current_joint_position()
        joint_positions = np.array(joint_positions)
        T_ee = self.panda.fkine(joint_positions)
        return np.array(T_ee)

    def wait_for_action_result(self, future, timeout: float = 30.0):
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        if not future.done():
            self.get_logger().warn(f"Action timed out after {timeout} seconds.")
            return None
        return future.result()

    def home(self) -> bool:
        self.get_logger().info("Moving to home position...")
        # current_joints = np.array(self.get_current_joint_position())
        # error = self.home_joints - current_joints
        # if np.linalg.norm(error) < self.goal_tolerance:
        #     return True
        # home_pose = self.panda.fkine(np.array(self.home_joints)).A @ self.T_tcp_link8.as_matrix()
        # self.goto_pose(Transform.from_matrix(home_pose))
        
        # while self.baseline_ready is False:
        #     time.sleep(0.1)
        # if cur_pose[2, 3] < 0.12:
        #     cur_pose[2, 3] += 0.08
        #     cur_pose = Transform.from_matrix(cur_pose)
        #     self.approach_grasp(cur_pose)
            # print('Lifted EE Z')
        
        # return self.goto_pose(Transform.from_matrix(home_pose))
        return self.goto_joints(self.home_joints)
    
    def set_table(self, table_height: float = 0., table_size: float = 0.5, T_base_task: Transform = Transform.identity()) -> None:
        pose = reorder_pose_list(T_base_task.to_list())
        pose[2] += table_height
        
        # 仅更新本地 Config 字典
        self.table_cfg = {
            "cuboid": {
                "table": {
                    "dims": [table_size, table_size, 0.01],
                    "pose": pose,
                },
            },
        }

    # --- 核心修改：Planning ---
    def planning(self, target_pose, pcl=None, plan_config=None):
        """
        调用 ZMQ Client 进行规划，支持 PCL 输入
        """
        target_pose = reorder_pose_list((target_pose * self.T_tcp_link8).to_list())
        
        # 准备数据
        # 1. Cuboids (Table)
        cuboids_data = self.table_cfg # 默认为 {'cuboid': {'table': ...}}

        # 2. PointCloud (如果你有)
        # pcl 必须是 (N, 3) 的 numpy 数组
        pcl_data = pcl 

        # 发送更新请求给 Server
        # Server 会在 GPU 上处理 Mesh 生成
        self.planner.update_world(cuboids=cuboids_data, pcl=pcl_data)
        
        # 请求规划
        cur_joint = np.array(self.get_current_joint_position())
        joint_waypoints, success = self.planner.plan(cur_joint, target_pose, plan_config=plan_config)
        
        return joint_waypoints


    # def goto_joints(self, target_joints):
    #     # (逻辑不变，使用 self.kp, self.kd 控制)
    #     arrived = False
    #     current_joints = np.array(self.get_current_joint_position())
    #     error = target_joints - current_joints
    #     self.react_control_flag = True
    #     last_velocity = np.array([0.0]*7)
    #     last_error = error 
        
    #     if np.linalg.norm(error) < self.goal_tolerance:
    #         self.react_control_flag = False
    #         return True
            
    #     while np.linalg.norm(error) > self.goal_tolerance and (self.robot_error is False):
    #         loop_start = time.time()
    #         robot_state_joint = self.get_current_joint_position()
    #         error = target_joints - np.array(robot_state_joint)
    #         error_derivative = (error - last_error) / self.dt
    #         raw_joint_vel = (self.kp * error) + (self.kd * error_derivative)
            
    #         joint_acc = (raw_joint_vel - last_velocity) / self.dt
    #         if (np.abs(joint_acc) > self.limit_acc).any():
    #             joint_acc = np.clip(joint_acc, -self.limit_acc, self.limit_acc)
    #             joint_vel = last_velocity + joint_acc * self.dt
    #         else:
    #             joint_vel = np.clip(raw_joint_vel, -self.limit_vel, self.limit_vel)
            
    #         last_velocity = joint_vel
    #         last_error = error
    #         self.joint_command_msg.name = self.joint_names
    #         self.joint_command_msg.velocity = joint_vel.tolist()
    #         elapsed = time.time() - loop_start
    #         if self.dt > elapsed:
    #             time.sleep(self.dt - elapsed)
    #     self.react_control_flag = False
    #     return True

    def goto_joints(self, target_joints, inter_points=20):
        current_joints = np.array(self.get_current_joint_position())
        interpolated_traj = rtb.jtraj(current_joints, target_joints, inter_points)
        if 'velocity' in self.control_topic:
            return self.goto_joint_trajectory(interpolated_traj.q)
        return self.goto_joint_trajectory_pos(interpolated_traj.q)
    

    def approach_grasp(self, target_pose, Gain=1, threshold=0.01, duration_limit=5.0):
        # target_pose = target_pose * self.T_tcp_body
        target_pose = target_pose * self.T_link8_tcp
        self.react_control_flag = True
        wrench_baseline = self.wrench_filtered.copy()
        k_null = 0.1
        dq = np.zeros((7, 1))
        last_velocity = np.zeros(7)
        start_time = time.time()
        while (self.robot_error is False):
            if np.linalg.norm(self.wrench_filtered-wrench_baseline) > self.force_limit:
                print('Force limit reached, stopping reactive control.')
                break
            if (time.time() - start_time) > duration_limit:
                print(f"[Approach] Time limit {duration_limit}s reached, stopping.")
                break
            q = np.array(self.get_current_joint_position())
            Te = self.panda.fkine(q)
            Tep = target_pose.as_matrix()
            v, arrived = rtb.p_servo(Te, Tep, Gain, threshold)
            if arrived:
                break
            jacobian = self.panda.jacobe(q)
            jacobian_pinv = np.linalg.pinv(jacobian)
            dq_task = jacobian_pinv @ v
            I = np.eye(jacobian.shape[1])  # 7x7 单位矩阵
            null_space_projection = I - (jacobian_pinv @ jacobian)
            dq_0 = -dq # 阻尼效果
                
            dq_null = null_space_projection @ dq_0
            
            # 4. 合并总速度
            raw_joint_vel = dq_task + k_null * dq_null.reshape(-1)
            dq = raw_joint_vel.reshape(-1, 1)

            # ---------- 4. 加速度/速度限制 ----------
            joint_acc = (raw_joint_vel - last_velocity) / self.dt
            joint_acc = np.clip(joint_acc, -self.limit_acc, self.limit_acc)
            joint_vel = last_velocity + joint_acc * self.dt
            joint_vel = np.clip(joint_vel, -self.limit_vel, self.limit_vel)

            last_velocity = joint_vel

            self.joint_command_msg.velocity = joint_vel.tolist()
        # ---------- 5. 缓停 ----------
        for _ in range(5):
            joint_vel *= 0.5
            self.joint_command_msg.velocity = joint_vel.tolist()
            time.sleep(0.01)

        self.react_control_flag = False
        return


    def goto_pose_reactive(self, target_pose, Gain=1, Lambda=0.1, threshold=0.001, detect_force=True, pcl=None, watch_dog_limit=70):
        # target_pose = target_pose * self.T_tcp_body
        target_pose = target_pose * self.T_link8_tcp
        self.react_control_flag = True
        arrived = False
        watch_dog = 0
        last_joint = np.array(self.get_current_joint_position())
        last_velocity = np.array([0.0]*7)
        joint_vel = np.array([0.0]*7)
        wrench_baseline = self.wrench_filtered.copy()
        while True and (self.robot_error is False):
            loop_start = time.time()
            robot_state_joint = self.get_current_joint_position()
            # Gain_mod = Gain * max((self.force_limit - np.linalg.norm(self.wrench)) / (self.force_limit - 6), 0)
            # print('Gain mod:', Gain_mod)
            raw_joint_vel, arrived = calculate_velocity(self.panda, np.array(robot_state_joint), target_pose, 
                                                    obstacles=None, Gain=Gain, Lambda=Lambda, threshold=threshold)
            joint_movement = np.linalg.norm(np.array(self.get_current_joint_position())-last_joint)
            last_joint = np.array(self.get_current_joint_position())
            if joint_movement < 0.001:
                watch_dog += 1
            else:
                watch_dog = 0
            if arrived is True or watch_dog > watch_dog_limit:
                break
            if np.linalg.norm(self.wrench_filtered-wrench_baseline) > self.force_limit and detect_force:
                print('Force limit reached, stopping reactive control.')
                break
            
            joint_acc = (raw_joint_vel - last_velocity) / self.dt
            joint_acc = np.clip(joint_acc, -self.limit_acc, self.limit_acc)
            joint_vel = last_velocity + joint_acc * self.dt
            joint_vel = np.clip(joint_vel, -self.limit_vel, self.limit_vel)
            
            self.joint_command_msg.velocity = joint_vel.tolist()
            last_velocity = joint_vel
            elapsed = time.time() - loop_start
            if self.dt > elapsed:
                time.sleep(self.dt - elapsed)
        
        for _ in range(5):
            joint_vel = joint_vel*0.5
            self.joint_command_msg.velocity = (joint_vel).tolist()
            time.sleep(0.01)
        self.react_control_flag = True
        return arrived
    
    # def grasp(self, width=0.01, speed=0.1, force=50.0, 
    #       inner_epsilon=0.005, outer_epsilon=0.005,
    #       hold=True, hold_interval=0.2):

    #     if not self.grasp_client.wait_for_server(timeout_sec=5.0):
    #         return False

    #     goal = Grasp.Goal()
    #     goal.width = width
    #     goal.speed = speed
    #     goal.force = force
    #     goal.epsilon = GraspEpsilon(inner=inner_epsilon, outer=outer_epsilon)

    #     send_future = self.grasp_client.send_goal_async(goal)
    #     result = self.wait_for_action_result(send_future, 5.0)
    #     if not result:
    #         return False

    #     # --- 关键：持续保持 ---
    #     if hold:
    #         final_width = width
    #         for _ in range(5):  # 初期补偿多几次
    #             self.move_gripper(final_width, speed=0.02, force=force)
    #             time.sleep(hold_interval)

    #     return True
    
    def move_gripper(self, position: float, max_effort: float = 50.0, timeout: float = 5.0) -> bool:
        """
        Move Robotiq gripper to specified position
        
        Args:
            position: Target position (0.0 = fully open, ~0.8 = fully closed, adjust based on your gripper)
            max_effort: Maximum effort in Newtons
            timeout: Action timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Robotiq gripper action server not available")
            return False
        
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        
        send_future = self.gripper_client.send_goal_async(goal)
        
        start = time.time()
        while not send_future.done() and (time.time() - start) < timeout:
            time.sleep(0.01)
            
        if not send_future.done():
            self.get_logger().warn("Gripper command timed out")
            return False
            
        handle = send_future.result()
        if not handle or not handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return False
            
        res_future = handle.get_result_async()
        res = self.wait_for_action_result(res_future, timeout)
        
        return res is not None and res.status == 4

    def grasp(self, position: float = 1.0, max_effort: float = 50.0) -> bool:
        """
        Close gripper to grasp object
        
        Args:
            position: Closing position (higher value = more closed, typically 0.6-0.8)
            max_effort: Gripping force in Newtons
            
        Returns:
            True if successful
        """
        self.get_logger().info(f"Grasping with position={position}, effort={max_effort}")
        return self.move_gripper(position, max_effort)

    def release(self, position: float = 0.0, max_effort: float = 50.0) -> bool:
        """
        Open gripper to release object
        
        Args:
            position: Opening position (0.0 = fully open)
            max_effort: Opening force in Newtons
            
        Returns:
            True if successful
        """
        self.get_logger().info(f"Releasing gripper to position={position}")
        return self.move_gripper(position, max_effort)

    def goto_pose(self, target_pose: Transform, pcl=None, plan_config=None) -> bool:
        cur_pose = Transform.from_matrix(self.get_ee_pose()) * self.T_tcp_link8.inverse()
        ## whether arrive at target
        dist =  (target_pose * cur_pose.inverse())
        if np.linalg.norm(dist.translation) < 0.01 and dist.rotation.as_euler('ZYX').abs().sum() < 0.1:
            return True
        
        self.get_logger().info(f"Planning to target pose...")
        joint_waypoints = self.planning(target_pose, pcl=pcl, plan_config=plan_config)
        if joint_waypoints is None:
            self.get_logger().error("Planning failed.")
            return False
        self.visualize_trajectory(joint_waypoints, self.panda)
        return self.goto_joint_trajectory_pos(joint_waypoints)
    
    def goto_joint_trajectory_pos(self, joint_waypoints: List[List[float]], alignment=True, non_blocking=False, replace_queue=True) -> bool:
        ok = self.set_joint_trajectory_queue(joint_waypoints, replace=replace_queue, alignment=alignment)
        if not ok:
            return False
        if non_blocking:
            return True

        timeout_start = time.time()
        timeout = max(3.0, len(joint_waypoints) * self.dt * 4.0 + (3.0 if alignment else 0.0))
        while self.robot_error is False and (time.time() - timeout_start) < timeout:
            if not self.is_trajectory_running():
                return True
            time.sleep(min(self.dt, 0.01))

        return not self.is_trajectory_running()

    def goto_joint_trajectory(self, joint_waypoints: List[List[float]]) -> bool:
        waypoints = np.array(joint_waypoints)
        if len(waypoints) < 2: return False
        self.react_control_flag = True
        
        process_tolerance = 0.4
        final_tolerance = self.goal_tolerance
        last_velocity = np.array([0.0] * 7)
        last_error = np.zeros(7)
        cur_i = 0
        while cur_i < len(waypoints) - 1 and self.robot_error is False:
            loop_start = time.time()
            cur_target = waypoints[cur_i + 1]
            current_joints = np.array(self.get_current_joint_position())
            error = cur_target - current_joints
            if np.linalg.norm(error) < (process_tolerance if cur_i < len(waypoints)-2 else process_tolerance*0.5):
                cur_i += 1
            error_derivative = (error - last_error) / self.dt
            raw_joint_vel = (self.kp * error) + (self.kd * error_derivative)
            
            joint_acc = np.clip((raw_joint_vel - last_velocity) / self.dt, -self.limit_acc, self.limit_acc)
            joint_vel = np.clip(last_velocity + joint_acc * self.dt, -self.limit_vel, self.limit_vel)
            
            last_velocity = joint_vel
            last_error = error
            self.joint_command_msg.name = self.joint_names
            self.joint_command_msg.velocity = joint_vel.tolist()
            if self.dt > (time.time() - loop_start): time.sleep(self.dt - (time.time() - loop_start))
        
        # Final Alignment
        timeout_start = time.time()
        final_target = waypoints[-1]
        while self.robot_error is False and (time.time() - timeout_start < 3.0):
            current_joints = np.array(self.get_current_joint_position())
            error = final_target - current_joints
            if np.linalg.norm(error) < final_tolerance: break
            joint_vel = np.clip(self.kp * error, -0.2, 0.2)
            self.joint_command_msg.velocity = joint_vel.tolist()
            time.sleep(self.dt)
        
        self.joint_command_msg.velocity = [0.0]*7
        time.sleep(0.05)
        self.react_control_flag = False
        return True

    def recover(self) -> bool:
        if not self.error_recovery_client.wait_for_server(timeout_sec=5.0): return False
        goal = ErrorRecovery.Goal()
        future = self.error_recovery_client.send_goal_async(goal)
        start = time.time()
        while not future.done() and (time.time() - start) < 5.0: time.sleep(0.01)
        if not future.done(): return False
        handle = future.result()
        if not handle or not handle.accepted: return False
        res = self.wait_for_action_result(handle.get_result_async(), 5.0)
        if res and res.status == 4:
            return self.home()
        return False

# --- Pose Generator (保留用于测试) ---
class FrankaPoseGenerator:
    def __init__(self, robot_name="fr3"):
        self.panda = rtb.models.Panda()
        self.q_min = np.array([-2.7, -1.7, -2.8, -3.0, -2.8, 0.01, -2.8]) * 0.5
        self.q_max = np.array([ 2.7,  1.7,  2.8, -0.1,  2.8, 3.7,  2.8]) * 0.5
    
    def _to_user_format(self, pos, quat):
        user_quat = [quat[3], quat[0], quat[1], quat[2]] 
        return Transform.from_list(user_quat + list(pos))

    def get_guaranteed_reachable_pose(self):
        q_rand = np.random.uniform(self.q_min, self.q_max)
        T = self.panda.fkine(q_rand)
        pos = T.t 
        quat = Rotation.from_matrix(T.R).as_quat()
        if pos[2] < 0.05: return self.get_guaranteed_reachable_pose()
        return self._to_user_format(pos, quat)

def main():
    rclpy.init()
    panda_commander = PandaCommander(robot_name="fr3")
    from rclpy.executors import MultiThreadedExecutor
    from threading import Thread
    executor = MultiThreadedExecutor(10)
    executor.add_node(panda_commander)
    generator = FrankaPoseGenerator()
    try:
        t1 = Thread(target=executor.spin, daemon=True)
        t1.start()

        for i in range(10):
            print(f"--- Loop {i} ---")
            print("Testing Gripper Open...")
            panda_commander.release(position=0.0, max_effort=50.0)
            # time.sleep(1.0)
            
            print("Testing Home...")
            panda_commander.home()
            # time.sleep(1.0)
            
            for j in range(2):
                print(f"Testing Plan to Random Pose {j}...")
                target_pose = generator.get_guaranteed_reachable_pose()
                panda_commander.goto_pose(target_pose)
                time.sleep(1.0)
            
            # print("Testing Gripper Close/Grasp...")
            panda_commander.grasp(position=0.8, max_effort=50.0)
            # time.sleep(1.0)
        
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        panda_commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()