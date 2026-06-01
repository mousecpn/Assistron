import time
import pickle
import zmq
import numpy as np
import cv2
import threading
from queue import Queue
import sys
sys.path.append('/home/u0161364/pi05_deploy')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# 假设依赖包在你的环境中存在
from openpi_client import websocket_client_policy, image_tools
from panda_control import PandaCommander 
from rclpy.executors import MultiThreadedExecutor
from utils.joy_listener import JoyListener
from sensor_msgs.msg import Joy
from utils.control import velocity_based_control
from rclpy.callback_groups import ReentrantCallbackGroup
# from miscellaneous.intent_client import IntentClient
from pi05_client import pi05_client


prompt = "open the drawer, and put the grape in the drawer"


class pi05_deploy(Node):
    def __init__(self, logger):
        super().__init__('pi05_deploy')
        self.logger = logger
        
        self.bridge = CvBridge()
        self.pc = PandaCommander() # 初始化 PandaCommander 实例
        self.client = pi05_client()
        # self.intent_client = IntentClient()
        self.callback_group = ReentrantCallbackGroup()

        # --- 传感器数据容器 ---
        self.left_img = None
        self.wrist_img = None
        self.gripper_width = 0.0
        self.gripper_state = 'open'

        # --- ROS 订阅器 ---
        self.gripper_sub = self.create_subscription(JointState, '/robotiq/joint_states', self.gripper_callback, 1, callback_group=self.callback_group)
        self.left_cam_sub = self.create_subscription(Image, '/left_camera/color/image_raw', self.left_camera_callback, 1, callback_group=self.callback_group)
        self.wrist_cam_sub = self.create_subscription(Image, '/wrist_camera/color/image_raw', self.wrist_camera_callback, 1, callback_group=self.callback_group)

        # ==========================================
        # RTC (Real-Time Chunking) 算法核心参数初始化
        # ==========================================
        self.H = 15                  # Prediction horizon (模型预测的总步数)
        self.dt = 1.0 / 15.0         # Controller timestep (控制器的时间间隔, 15Hz)
        self.s_min = 5               # Minimum execution horizon (强制的最短执行步数)
        self.b = 10                  # Delay buffer size (历史延迟队列长度)
        self.action_quat = 1
        
        # 维护一个历史延迟队列 Q，用来预估下一次的 d
        self.Q = Queue(maxsize=self.b)
        self.d_init = 3
        self.Q.put(self.d_init)      

        self.A_cur = None            # 当前正在执行的 Action Chunk
        self.mutex = threading.Lock()# 保护共享变量的互斥锁

        self.joy_listener = JoyListener()
        self.joy_subscriber = self.create_subscription(
            Joy,
            '/joy',
            self.joy_listener.update_from_joy_msg,
            1,
            callback_group=self.callback_group
        )

        self.last_joint_position = None
        self.last_execution_time = None

    def gripper_callback(self, msg):
        try:
            index1 = msg.name.index('robotiq_85_left_knuckle_joint')
            width1 = msg.position[index1]
            self.gripper_width = width1
            self.gripper_state = 'open' if width1 == 0.0 else 'close'
        except ValueError:
            pass
    
    def process_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge 转换错误: {e}")
            return None

    def wrist_camera_callback(self, msg):
        wrist_img = self.process_image(msg)
        if wrist_img is not None:
            wrist_img = cv2.rotate(wrist_img, cv2.ROTATE_180)
            wrist_img = image_tools.resize_with_pad(wrist_img, 224, 224)
            self.wrist_img = wrist_img
    
    def left_camera_callback(self, msg):
        left_img = self.process_image(msg)
        if left_img is not None:
            left_img = image_tools.resize_with_pad(left_img, 224, 224)
            self.left_img = left_img

    def inference_loop(self):
        """
        后台连续推理循环 (对应论文 Algorithm 1: INFERENCELOOP)
        """
        self.get_logger().info("RTC 异步推理线程已启动...")
        joint_positions = None
        joint_positions_prev = None
        gripper_flag = False
        start_teleop = False
        timemapping = None
        vla_flag = False

        # ✅ 1. 新增：定义一个标志位，记录是否已经打印过等待信息
        waiting_printed = False

        while rclpy.ok():
            if not waiting_printed:
                # print("waiting for valid joy input and sensor data... at time:", time.time())
                waiting_printed = True  # 打印后立刻设为 True，阻止后续循环重复打印
            # time.sleep(0.01)
            twist = self.joy_listener.get_twist_array() 
            if twist is None or np.linalg.norm(twist) < 0.01: # or self.H - len(self.pc._trajectory_queue) < self.s_min + 1:
                if self.last_execution_time is not None and time.time() - self.last_execution_time > 0.01:
                    self.stop()
                    # print("检测到长时间未执行，已发送停止指令，等待新的有效输入...")
                time.sleep(0.01)
                continue

            if self.pc.robot_error:
                time.sleep(0.1)
                continue
            
            if self.left_img is None or self.wrist_img is None:
                time.sleep(0.01)
                continue
            t_loop_start = time.time()
            
            waiting_printed = False
            
            if not start_teleop and np.linalg.norm(twist) > 0.001:
                start_teleop = True
                self.logger.info("[Logging] Starting teleoperation...")
                start_time = time.time()
            
            vx, vy, vz, wx, wy, wz, home, grasp, release, approach_grasp, drop = twist
            pos_command = np.array([vx, vy, vz]) #+ 0.05 * np.random.randn(3) # 添加小幅随机扰动，增加输入的多样性，促进模型泛化
            ang_command =  np.array([wx, wy, wz]) #+ 0.05 * np.random.randn(3) # 添加小幅随机扰动，增加输入的多样性，促进模型泛化
            velocity_command = np.concatenate((pos_command, ang_command))


            if self.gripper_state == 'open' and grasp > 0.5:
                gripper_flag = True
                gripper_command = 0.8
            elif self.gripper_state == 'close' and release > 0.5:
                gripper_flag = True
                gripper_command = 0.0
            else:
                gripper_command = self.gripper_width


            if home == 1:
                if start_teleop:
                    start_teleop = False
                    self.logger.info("[Logging] Task Finished.")
                    result = input("Please enter the progress: ")
                    self.logger.info(f"[Logging] progress: {result}")
                with self.mutex:
                    self.pc.home()
                    self.stop()
                    continue
            


            with self.mutex:
                if self.A_cur is None:
                    # 如果系统刚启动或被重置，强制进度达标以立即触发推理
                    s_current = self.s_min 
                else:
                    # 💡 核心改进：H 减去底层队列中还未执行的动作数，就是当前 Chunk 已经执行了的动作数
                    s_current = self.H - len(self.pc._trajectory_queue) * self.action_quat

            
            # grasp as a trigger for immediate inference 
            if approach_grasp < 0.5:
                gripper_flag = self.grasp(gripper_command)
                if gripper_flag:
                    continue # 如果执行了抓取动作，立即进入下一轮循环，重新评估状态并触发推理

                
            if s_current < self.s_min:
                time.sleep(0.01) # 还没执行够 s_min 步，继续等
                continue
            
            with self.mutex:
                s = int(s_current) # 记录此时准确的已执行步数
                
                if self.A_cur is not None:
                    # 截取当前 Chunk 尚未执行完的剩余部分
                    A_prev = self.A_cur[s:self.H]
                    # 提取历史队列中最悲观（最大）的延迟作为下一次 d 的预估
                    d_estimated = max(list(self.Q.queue)) 
                else:
                    A_prev = None
                    d_estimated = 0
                if len(self.pc._trajectory_queue) == 0:
                    d_estimated = 0  # 如果底层队列空了，说明动作已经完全执行了，预估延迟为0
                
            t_setup_end = time.time()
            
            if joint_positions is not None:
                # self.pc.visualize_last_trajectory(joint_positions, self.pc.panda)
                joint_positions_prev = joint_positions.copy()

            # 记录推理前的硬件队列长度
            queue_len_before = len(self.pc._trajectory_queue) * self.action_quat

            # 获取最新的图像与物理状态
            t_preproc_start = time.time()
            left_img = self.left_img
            wrist_img = self.wrist_img

            
            cur_q = np.array(self.pc.get_current_joint_position())
            cur_gripper_position = self.gripper_width
            t_preproc_end = time.time()
    
            t_vel_ctrl_start = time.time()
            command_joint_velocity = velocity_based_control(self.pc.panda, cur_q, pos_command, ang_command,  onbase=True)
            command_joint_velocity = np.repeat(command_joint_velocity.reshape(1, -1), self.H, axis=0) * 2 # 扩展到整个预测 horizon
            # decaying_factors = np.linspace(1.0, 0.1, self.H)[:, None]
            decaying_factors = np.array([0.8**t for t in range(self.H)])[:, None]  # 指数衰减，近期命令权重更高
            command_joint_velocity = command_joint_velocity * decaying_factors  # 应用衰减
            t_vel_ctrl_end = time.time()

            t_proposed_start = time.time()
            # 构造 proposed_action (将 A_prev 右侧填充对齐到 H 长度)
            if approach_grasp < 0.5:
                if A_prev is not None:
                    proposed_action = np.zeros((self.H, 32), dtype=np.float32)
                    proposed_action[:len(A_prev), :8] = A_prev
                    # proposed_action[d_estimated:, :7] = np.repeat(command_joint_velocity[None, :], proposed_action[d_estimated:, :7].shape[0], axis=0)
                    proposed_action[d_estimated:, :7] = command_joint_velocity[:self.H - d_estimated]
                    proposed_action[d_estimated:, 7] = gripper_command
                else:
                    proposed_action = None
            else:
                if A_prev is not None:
                    proposed_action = np.zeros((self.H, 32), dtype=np.float32)
                    proposed_action[:len(A_prev), :8] = A_prev
                else:
                    proposed_action = None
            
            ## integrate proposed action
            proposed_joint_position = np.zeros((self.H - d_estimated, 7), dtype=np.float32)
            proposed_joint_position[0] = cur_q + command_joint_velocity[0] * self.dt
            for t in range(1, self.H - d_estimated):
                proposed_joint_position[t] = proposed_joint_position[t-1] + command_joint_velocity[t] * self.dt
            t_proposed_end = time.time()


            t_viz_start = time.time()
            if joint_positions is not None and proposed_action is not None:
                self.pc.visualize_last_trajectory(proposed_joint_position, self.pc.panda)
            
            

            print(f"🎯 当前用户意图: {prompt}")
            print(f"📊 当前执行进度 s: {s}, 历史延迟预估 d: {d_estimated}")
            input_data = {
                "observation/exterior_image_1_left": left_img,
                "observation/wrist_image_left": wrist_img,
                "observation/joint_position": cur_q,
                "observation/gripper_position": np.array([float(self.gripper_state == 'close')], dtype=np.float32),
                "prompt": prompt,
                "proposed_action": proposed_action,
                "velocity_command": velocity_command[None, :].repeat(self.H, axis=0),
                "d": d_estimated,
                "s": s,
            }

            self.client.update_data(input_data)
            t_viz_end = time.time()
            
            t_inference_start = time.time()
            action_new = self.client.send_request() 
            t_inference_end = time.time()
            
            if action_new is None:
                continue # 网络失败则重试

            t_post_inf_start = time.time()
            with self.mutex:
                # try:
                queue_len_after = len(self.pc._trajectory_queue) * self.action_quat
                observed_delay = queue_len_before - queue_len_after
                # print(f"📈 实际观察到的延迟 (以动作步数计): {observed_delay}")

                # except:
                #     observed_delay = 0
                
                # 防御性截断，避免因外部干扰导致数值为负
                observed_delay = max(0, observed_delay)
                
                # 登记真实延迟到队列中
                if self.Q.full():
                    self.Q.get()
                self.Q.put(observed_delay)

                # 正式接管，刷新 A_cur 为新一代 chunk
                self.A_cur = action_new.copy()
            
            
            

            t_post_inf_end = time.time()
            start_idx = min(observed_delay, self.H - 1)
            if approach_grasp > 0.5:
                gripper_command = self.A_cur[start_idx+1, 7] if self.A_cur is not None else cur_gripper_position
                gripper_flag = self.grasp(gripper_command)
                if gripper_flag:
                    continue

            ## if the acceleration is too large, execute the proposed action without blending
            t_blending_start = time.time()
            max_acceleration = self.acceleration_calculation(self.A_cur[start_idx:, :7])
            # print(f"⚠️ 预测动作的最大加速度 {max_acceleration:.2f} ")
            # if max_acceleration > 300.0 and proposed_action is not None:
            #     remaining_actions = command_joint_velocity[:self.H-start_idx, :7]
            # else:
            #     remaining_actions = self.A_cur[start_idx:, :7]
            #     if approach_grasp < 0.5:
            #         remaining_actions, vla_flag = self.policy_blending(remaining_actions, command_joint_velocity[:self.H-start_idx])  # 可选的策略融合，增强人机协同
            t_blending_end = time.time()
            # if not vla_flag:
            #     start_idx = 0
            remaining_actions = self.A_cur[start_idx:, :7]
            # self.logger.info(f"actions from model: {self.A_cur[start_idx:,:7].flatten().tolist()}")
            t_integration_start = time.time()
            if len(remaining_actions) > 0:
                # 重新对齐物理世界：从机器人当前真实位置开始积分
                joint_positions = np.zeros((min(self.H, start_idx + len(remaining_actions)), 7))
                cur_j = np.array(self.pc.get_current_joint_position())
                if gripper_flag is False and joint_positions_prev is not None:  
                    # 如果动作很小且之前有轨迹，则从上次轨迹的末尾继续积分
                    idx = np.argmin(np.linalg.norm(joint_positions_prev- cur_j[None, :], axis=1))
                    error = np.linalg.norm(cur_j-joint_positions_prev[idx])
                    if error < 0.05:
                        # print("从上次轨迹末尾继续积分, error:", error, "step:", idx)
                        cur_j = joint_positions_prev[idx]

                joint_positions[start_idx] = cur_j + remaining_actions[0] * self.dt
                for i in range(1, len(remaining_actions)):
                    joint_positions[start_idx + i] = joint_positions[start_idx + i-1] + remaining_actions[i] * self.dt
                
                if start_idx >= 1:
                    # reverse integration
                    for i in range(start_idx-1, -1, -1):
                        joint_positions[i] = joint_positions[i+1] - self.A_cur[i][:7] * self.dt
                
                exe_joint_positions = joint_positions[start_idx+2:]
                exe_joint_positions = self.truncate_trajectory(exe_joint_positions, remaining_actions[2:], max_displacement=np.linalg.norm(command_joint_velocity[0, :7]* 0.6)) # 根据当前命令速度动态调整截断阈值
                t_integration_end = time.time()

                t_traj_exec_start = time.time()
                self.pc.visualize_trajectory(exe_joint_positions, self.pc.panda)

                self.pc.goto_joint_trajectory_pos(
                    exe_joint_positions,
                    alignment=False,
                    non_blocking=True,
                    replace_queue=True
                )
                t_traj_exec_end = time.time()

                cur_joint_position = np.array(self.pc.get_current_joint_position())
                if self.last_joint_position is not None and np.linalg.norm(cur_joint_position - self.last_joint_position) > 0.01:
                    self.last_execution_time = time.time()
                elif self.last_joint_position is None:
                    self.last_execution_time = time.time()
                self.last_joint_position = cur_joint_position

                t_loop_end = time.time()
                # print(
                #     f"⏱️  Loop breakdown (ms) | "
                #     f"setup: {(t_setup_end - t_loop_start)*1000:.1f} | "
                #     f"preproc: {(t_preproc_end - t_preproc_start)*1000:.1f} | "
                #     f"vel_ctrl: {(t_vel_ctrl_end - t_vel_ctrl_start)*1000:.1f} | "
                #     f"proposed: {(t_proposed_end - t_proposed_start)*1000:.1f} | "
                #     f"viz+prep: {(t_viz_end - t_viz_start)*1000:.1f} | "
                #     f"inference: {(t_inference_end - t_inference_start)*1000:.1f} | "
                #     f"post_inf: {(t_post_inf_end - t_post_inf_start)*1000:.1f} | "
                #     f"blending: {(t_blending_end - t_blending_start)*1000:.1f} | "
                #     f"integration: {(t_integration_end - t_integration_start)*1000:.1f} | "
                #     f"traj_exec: {(t_traj_exec_end - t_traj_exec_start)*1000:.1f} | "
                #     f"tail: {(t_loop_end - t_traj_exec_end)*1000:.1f} | "
                #     f"total: {(t_loop_end - t_loop_start)*1000:.1f}"
                # )
                cur_pose = self.pc.get_ee_pose()
                self.logger.info(f'[Logging] Current pose: {cur_pose.flatten().tolist()}')
    
    def truncate_trajectory(self, joint_positions, joint_velocities, max_displacement: float = 0.5):
        """Truncate trajectory once cumulative |dq * dt| exceeds max_displacement (radians).

        joint_velocities: (N, 7) array aligned with joint_positions.
        """
        cumulative = 0.0
        for i in range(len(joint_velocities)):
            cumulative += np.linalg.norm(joint_velocities[i]) * self.dt
            if cumulative > max_displacement:
                return joint_positions[:i + 1]
        return joint_positions
    
    def acceleration_calculation(self, proposed_action):
        # 计算加速度
        acceleration = np.diff(proposed_action[:, :7], n=2, axis=0) / (self.dt ** 2)
        # 计算加速度的范数
        try:
            acc_magnitude = np.linalg.norm(acceleration, axis=1)
            max_acceleration = np.max(acc_magnitude)
        except:
            max_acceleration = 0

        return max_acceleration

    def cosine_similarity(self, a, b):
        return np.einsum('ij,ij->i', a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    
    def policy_blending(self, prediction, command_joint_velocity):
        """
        prediction: 来自大模型的预测动作 (H, 8)，包含关节速度和抓手指令
        command_joint_velocity: 基于当前状态和用户输入计算的命令式关节速度 (7,)
        """
        ### forward kinematics ###
        J = self.pc.panda.jacob0(np.array(self.pc.get_current_joint_position()))
        eef_prediction = prediction[:, :7] @ J.T
        eef_command = command_joint_velocity @ J.T

        ### calculate agreement ####
        # command_joint_velocity = np.repeat(command_joint_velocity.reshape(1, -1), len(prediction), axis=0)
        # gamma degraded
        gammas = np.array([0.9**i for i in range(len(prediction))])
        gammas = gammas / np.sum(gammas)  # 归一化权重
        agreement = self.cosine_similarity(eef_prediction[:, :6], eef_command[:, :6])
        agreement = np.sum(agreement * gammas[None, :])
        print(f"Agreement between proposed action and command: {agreement:.3f}")
        if agreement > 0.0:
            return prediction, True
        # elif agreement < 0.3 and agreement > 0.0:
        #     blend_velocity = command_joint_velocity * (1 - agreement) + prediction[:, :7] * agreement
        #     blend_velocity = np.concatenate((blend_velocity, prediction[:, 7:8]), axis=1)  # 保留大模型的抓手指令
        #     return blend_velocity
        else:
            blend_velocity = np.concatenate((command_joint_velocity, prediction[:, 7:8]), axis=1)  # 保留大模型的抓手指令
            return blend_velocity, False

    def stop(self):
        self.pc.clear_joint_trajectory_queue()
        self.pc.joint_command_msg.name = self.pc.joint_names
        self.pc.joint_command_msg.position = self.pc.get_current_joint_position()
    
    def chunk_velocity_calculation(self, pos_command, ang_command):
        cur_q = np.array(self.pc.get_current_joint_position())
        command_joint_velocity = []
        for t in range(self.H):
            joint_vel_t = velocity_based_control(self.pc.panda, cur_q, pos_command, ang_command,  onbase=True)
            cur_q = cur_q + joint_vel_t * self.dt
            command_joint_velocity.append(joint_vel_t)
        return np.array(command_joint_velocity)
    
    def grasp(self, gripper_command):
        pred_gripper = gripper_command
        gripper_flag = False
        target_gripper = 0.0
        
        if pred_gripper > 0.5 and self.gripper_state == 'open':
            gripper_flag = True
            target_gripper = 0.8
        elif pred_gripper <= 0.5 and self.gripper_state == 'close':
            gripper_flag = True
            target_gripper = 0.0
            
        if gripper_flag:
            # while len(self.pc._trajectory_queue) > 0:
            #     time.sleep(0.1) # 等旧轨迹跑完
            # time.sleep(0.05)
            self.stop()
            self.pc.grasp(target_gripper)
            # time.sleep(0.3)
            
            self.A_cur = None # 强制复位，下一帧立即重新触发推理
        return gripper_flag
                


def main(args=None):
    rclpy.init(args=args)

    import logging
    import os
    from datetime import datetime
    methods_dict = {1:'direct',
                    2:'pure language',
                    3:'Assistron'}
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    user_name = 'test'
    # if user_name == 'test':
    #     global ADD_NOISE
    #     ADD_NOISE = False
    os.makedirs("logs_ablation/{}".format(user_name), exist_ok=True)
    method_idx = 3
    method = 'posterior' # methods_dict[method_idx]
    log_file_path = os.path.expanduser(os.path.join("logs_ablation/{}".format(user_name), '_'.join((current_time,user_name,method))+".log"))
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    node = pi05_deploy(logger)
    
    executor = MultiThreadedExecutor(12)
    executor.add_node(node)
    executor.add_node(node.pc)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    node.pc.home()  # 机械臂回初始位置
    node.pc.release() # 确保抓手松开
    
    try:
        node.inference_loop()
    except KeyboardInterrupt:
        node.get_logger().info("程序被用户手动终止")
    finally:
        node.stop()
        rclpy.shutdown()

if __name__ == '__main__':  
    main()