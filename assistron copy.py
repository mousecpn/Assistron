import collections
import time

import numpy as np
import cv2
import threading
from queue import Queue
from enum import Enum

# =============================================================================
# Camera backend configuration
# =============================================================================
# False → subscribe to /left_camera and /wrist_camera ROS2 topics


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import JoyFeedback
from cv_bridge import CvBridge, CvBridgeError
from franka_msgs.action import ErrorRecovery

from openpi_client import image_tools
from panda_control import PandaCommander
from rclpy.executors import MultiThreadedExecutor
from utils.joy_listener import JoyListener
from sensor_msgs.msg import Joy
from utils.control import velocity_based_control
from rclpy.callback_groups import ReentrantCallbackGroup
from pi05_client import pi05_client
from utils.voice_transcript_client import VoiceRecorder
from interaction_detection.infer_dual_model import DualSuccessDetector
from utils.shared_control_status_windows import SharedControlStatusWindows

USE_PYREALSENSE = False

SERIAL_LEFT  = '047322071010'
SERIAL_WRIST = '309622300781'
CAM_W, CAM_H, CAM_FPS = 424, 240, 30

if USE_PYREALSENSE:
    import pyrealsense2 as rs



# =============================================================================
# FSM State Definition
# =============================================================================
class RobotState(Enum):
    STOP         = "stop"
    AUTO         = "auto"
    SHARED       = "shared_control"
    MANUAL       = "manual"




# =============================================================================
# Main deployment node
# =============================================================================
class Assistron(Node):
    def __init__(self, logger):
        super().__init__('assistron')
        self.logger = logger

        
        self.bridge = CvBridge()
        self.pc = PandaCommander()
        self.client = pi05_client()
        self.callback_group = ReentrantCallbackGroup()
        self.detector = DualSuccessDetector()


        # --- sensor data ---
        self.left_img = None
        self.wrist_img = None
        self.wrist_img_raw = None
        self.left_img_raw = None
        self.gripper_width = 0.0
        self.gripper_state = 'open'

        # --- ROS subscribers ---
        self.gripper_sub = self.create_subscription(
            JointState, '/robotiq/joint_states', self.gripper_callback, 1,
            callback_group=self.callback_group)
        self._js_watchdog_sub = self.create_subscription(
            JointState, '/joint_states', self._js_watchdog_cb, 1,
            callback_group=self.callback_group)

        if USE_PYREALSENSE:
            # Start RealSense capture thread instead of subscribing to ROS topics
            self._rs_stop_event = threading.Event()
            self._rs_thread = threading.Thread(
                target=self._realsense_capture_thread, daemon=True)
            self._rs_thread.start()
            self.get_logger().info(
                f"RealSense backend enabled "
                f"(left={SERIAL_LEFT}, wrist={SERIAL_WRIST})")
        else:
            self.left_cam_sub = self.create_subscription(
                Image, '/left_camera/color/image_raw', self.left_camera_callback, 1,
                callback_group=self.callback_group)
            self.wrist_cam_sub = self.create_subscription(
                Image, '/wrist_camera/color/image_raw', self.wrist_camera_callback, 1,
                callback_group=self.callback_group)

        # --- RTC core parameters ---
        self.H          = 15
        self.dt         = 1.0 / 15.0
        self.s_min      = 5
        self.b          = 10
        self.action_quat = 1

        self.Q      = Queue(maxsize=self.b)
        self.d_init = 3
        self.Q.put(self.d_init)

        self.A_cur  = None
        self.mutex  = threading.Lock()

        # --- joystick ---
        self.joy_listener = JoyListener()
        self._auto_btn_prev = False  # edge-detect for Y (auto) button
        self.joy_subscriber = self.create_subscription(
            Joy, '/joy', self.joy_listener.update_from_joy_msg, 1,
            callback_group=self.callback_group)

        self.last_joint_position  = None

        # --- FSM ---
        self.fsm_state = RobotState.MANUAL
        self.get_logger().info(f"FSM initialised → {self.fsm_state.value}")

        # --- voice ---
        self.voice_recorder = VoiceRecorder(server_url="http://127.0.0.1:43100")


        # --- Joystick rumble ---
        self._rumble_pub = self.create_publisher(JoyFeedback, '/joy/set_feedback', 1)

        # --- Controller watchdog ---
        self._last_js_stamp  = time.time()   # updated by /joint_states messages
        self._js_stale_sec   = 4.0           # seconds without JS → assume crash
        self._recovering     = False         # guard against concurrent recovery

        self.current_prompt = "stop"
        self._ui_lock = threading.Lock()
        self._ui_extra_fields = {}

        # --- detector history ---
        # Each entry: {'time': float, 'pred': int, 'prob_1': float, 'gripper_state': str}
        self._detector_history = collections.deque(maxlen=200)
        self.status_window = SharedControlStatusWindows(
            snapshot_provider=self.get_ui_snapshot,
            logger=self.get_logger(),
        )
        self.refine_flag = False

        # Cooldown: after leaving MANUAL → AUTO, block re-entry for N seconds
        self._intervention_cooldown_sec = 2.0
        self._last_intervention_exit_time = 0.0

        # Cooldown: after entering SHARED (from AUTO), block return to AUTO for N seconds
        self._shared_cooldown_sec = 1.0
        self._last_shared_enter_time = 0.0


        ## logging variable
        self.total_commands = 0



        self.joint_positions_prev = None
        self.last_execution_time = None
        


    def get_ui_snapshot(self):
        with self._ui_lock:
            return {
                "language_instruction": self.voice_recorder.prompt,
                "fsm_state": self.fsm_state.value.upper(),
                "extra_fields": dict(self._ui_extra_fields),
            }

    # ------------------------------------------------------------------
    # RealSense capture thread
    # ------------------------------------------------------------------
    def _realsense_capture_thread(self):
        """Background thread: continuously capture frames from both RealSense
        cameras and populate self.left_img / self.wrist_img (and *_raw)."""
        def _make_pipeline(serial: str) -> rs.pipeline:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
            pipeline.start(cfg)
            return pipeline

        try:
            pipe_left  = _make_pipeline(SERIAL_LEFT)
            pipe_wrist = _make_pipeline(SERIAL_WRIST)
        except Exception as exc:
            self.get_logger().error(
                f"[RealSense] Failed to open cameras: {exc}")
            return

        self.get_logger().info("[RealSense] Both cameras opened successfully.")
        align = rs.align(rs.stream.color)

        try:
            while not self._rs_stop_event.is_set():
                # --- left camera ---
                try:
                    frames_left = pipe_left.wait_for_frames(timeout_ms=200)
                    color_left  = frames_left.get_color_frame()
                    if color_left:
                        bgr = np.asanyarray(color_left.get_data())
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        self.left_img_raw = rgb
                        self.left_img = image_tools.resize_with_pad(rgb, 224, 224)
                except RuntimeError:
                    pass  # timeout – keep last frame

                # --- wrist camera ---
                try:
                    frames_wrist = pipe_wrist.wait_for_frames(timeout_ms=200)
                    color_wrist  = frames_wrist.get_color_frame()
                    if color_wrist:
                        bgr = np.asanyarray(color_wrist.get_data())
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        self.wrist_img_raw = rgb
                        rotated = cv2.rotate(rgb, cv2.ROTATE_180)
                        self.wrist_img = image_tools.resize_with_pad(
                            rotated, 224, 224)
                except RuntimeError:
                    pass  # timeout – keep last frame
        finally:
            pipe_left.stop()
            pipe_wrist.stop()
            self.get_logger().info("[RealSense] Cameras stopped.")

    # ------------------------------------------------------------------
    # Controller watchdog & recovery
    # ------------------------------------------------------------------
    def _js_watchdog_cb(self, msg):
        """Heartbeat: update timestamp every time /joint_states arrives."""
        self._last_js_stamp = time.time()

    def _get_stable_joint_position(self, tol: float = 0.001, interval: float = 0.01,
                                max_wait: float = 1) -> np.ndarray:
        """
        Poll joint positions until two consecutive readings differ by less than
        `tol` (L2 norm), or until `max_wait` seconds have elapsed.
        Returns the latest stable reading.
        """
        q_prev = np.array(self.pc.get_current_joint_position())
        deadline = time.time() + max_wait
        while time.time() < deadline:
            time.sleep(interval)
            q_cur = np.array(self.pc.get_current_joint_position())
            if np.linalg.norm(q_cur - q_prev) < tol:
                return q_cur
            q_prev = q_cur
        return q_prev

    def _do_recovery(self):
        """Daemon thread: wait for controller to resume, nudge joints inside limits, then home."""
        # FR3 hard joint limits with 0.1 rad safety margin
        FR3_Q_MIN = np.array([-2.6437, -1.6837, -2.8007, -2.9421, -2.7065,  0.0825, -2.7973]) *0.9
        FR3_Q_MAX = np.array([ 2.6437,  1.6837,  2.8007, -0.2518,  2.7065,  3.6525,  2.7973]) * 0.9
        self.stop()

        self.get_logger().error("Controller down — waiting for /joint_states to resume …")

        # --- 1. Wait until /joint_states is fresh again ---
        while True:
            time.sleep(0.2)
            if time.time() - self._last_js_stamp < self._js_stale_sec:
                break
        self.get_logger().info("/joint_states resumed — controller is back.")

        # --- 2. Send ErrorRecovery to clear any latched fault ---
        self.get_logger().info("Sending ErrorRecovery goal to clear latched faults …")
        if self.pc.error_recovery_client.wait_for_server(timeout_sec=5.0):
            goal = ErrorRecovery.Goal()
            future = self.pc.error_recovery_client.send_goal_async(goal)
            start = time.time()
            while not future.done() and (time.time() - start) < 10.0:
                time.sleep(0.05)
            if future.done() and future.result() and future.result().accepted:
                result_future = future.result().get_result_async()
                start = time.time()
                while not result_future.done() and (time.time() - start) < 10.0:
                    time.sleep(0.05)
                self.get_logger().info("ErrorRecovery goal completed.")
            else:
                self.get_logger().warn("ErrorRecovery goal rejected or timed out — continuing anyway.")
        else:
            self.get_logger().warn("ErrorRecovery server not available — continuing anyway.")

        while True:
            time.sleep(0.2)
            if time.time() - self._last_js_stamp < self._js_stale_sec:
                break

        # --- 3. Nudge joints inside limits if needed ---
        try:
            q_cur = self._get_stable_joint_position()
            q_safe = np.clip(q_cur, FR3_Q_MIN, FR3_Q_MAX)
            # if np.linalg.norm(q_safe - q_cur) > 0.005:
            self.get_logger().info(
                f"Nudging joints to safe pose: delta={np.round(q_safe - q_cur, 1)}")
            
            self.pc.goto_joints(q_safe.tolist(),  inter_points=100)
            # else:
            #     self.get_logger().info("Joints already within limits — skipping nudge.")
        except Exception as e:
            self.get_logger().warn(f"Nudge step failed: {e}")


        # --- 5. Release guard ---
        self._recovering = False
        self.get_logger().info("Recovery complete — FSM remains in STOP, resume manually.")
        self._transition(RobotState.MANUAL)

    # ------------------------------------------------------------------
    # RViz status visualisation
    # ------------------------------------------------------------------
    def _rumble(self, intensity: float = 0.4, duration: float = 0.2):
        """Send a short rumble to the joystick via /joy/set_feedback."""
        fb = JoyFeedback()
        fb.type      = JoyFeedback.TYPE_RUMBLE
        fb.id        = 0
        fb.intensity = float(intensity)
        self._rumble_pub.publish(fb)
        # schedule a stop after `duration` seconds
        threading.Timer(duration, self._rumble_stop).start()

    def _rumble_stop(self):
        fb = JoyFeedback()
        fb.type      = JoyFeedback.TYPE_RUMBLE
        fb.id        = 0
        fb.intensity = 0.0
        self._rumble_pub.publish(fb)


    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
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
        self.wrist_img_raw = self.process_image(msg)
        if self.wrist_img_raw is not None:
            wrist_img = cv2.rotate(self.wrist_img_raw, cv2.ROTATE_180)
            self.wrist_img = image_tools.resize_with_pad(wrist_img, 224, 224)

    def left_camera_callback(self, msg):
        self.left_img_raw = self.process_image(msg)
        if self.left_img_raw is not None:
            self.left_img = image_tools.resize_with_pad(self.left_img_raw, 224, 224)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prompt_is_stop(self):
        p = self.voice_recorder.prompt
        return 'stop' in p or 'Stop' in p

    def _has_user_command(self, twist):
        """True when the user is pushing any joystick axis (excluding voice btn)."""
        # twist layout: vx, vy, vz, wx, wy, wz, home, grasp, release, auto, voice
        return np.linalg.norm(twist[:6]) > 0.0001 or twist[6] > 0.5 or twist[7] > 0.5 or twist[8] > 0.5

    def _vla_wants_gripper_change(self, action_chunk, idx):
        """
        Transition (3): VLA predicts a gripper state change in the upcoming chunk.
        Returns True without executing the gripper command.
        """
        if action_chunk is None:
            return False
        # pred_gripper = action_chunk[idx, 7]
        pred_gripper = action_chunk[idx:, 7].mean()
        if pred_gripper > 0.5 and self.gripper_state == 'open':
            return True
        if pred_gripper <= 0.5 and self.gripper_state == 'close':
            return True
        return False

    def _transition(self, new_state):
        if new_state != self.fsm_state:
            # Cooldown guard: block AUTO → MANUAL within 2s of leaving MANUAL
            if (new_state == RobotState.MANUAL and
                    self.fsm_state == RobotState.AUTO and
                    time.time() - self._last_intervention_exit_time < self._intervention_cooldown_sec):
                remaining = self._intervention_cooldown_sec - (time.time() - self._last_intervention_exit_time)
                self.get_logger().info(
                    f"FSM: MANUAL blocked (cooldown {remaining:.1f}s remaining)")
                return
            self.get_logger().info(
                f"FSM: {self.fsm_state.value} → {new_state.value}")
            if (new_state == RobotState.MANUAL and self.fsm_state == RobotState.AUTO): # or (new_state == RobotState.AUTO and self.fsm_state == RobotState.MANUAL):
                self._rumble()
            if (self.fsm_state == RobotState.AUTO and new_state == RobotState.MANUAL) or (self.fsm_state == RobotState.MANUAL and new_state == RobotState.AUTO):
                self._last_intervention_exit_time = time.time()                
                
            if new_state == RobotState.MANUAL:
                self.stop()
            self.fsm_state = new_state
            self.logger.info(f'[Logging] FSM Transition')
            self.logger.info(f'[Logging] Current mode: {self.fsm_state.value.upper()}')
          
            
    def stop(self):
        self.pc.clear_joint_trajectory_queue()
        self.pc.joint_command_msg.name = self.pc.joint_names
        self.pc.joint_command_msg.position = self.pc.get_current_joint_position()

    def grasp(self, gripper_command):
        """Execute gripper command and reset action chunk."""
        gripper_flag = False
        target_gripper = gripper_command
        if gripper_command > 0.5 and self.gripper_state == 'open':
            gripper_flag = True
            target_gripper = 0.8
        elif gripper_command <= 0.5 and self.gripper_state == 'close':
            gripper_flag = True
            target_gripper = 0.0
        if gripper_flag:
            # while len(self.pc._trajectory_queue) > 0:
            #     time.sleep(0.1)
            self.stop()
            self.A_cur = None
        return target_gripper, gripper_flag

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
        


    # ------------------------------------------------------------------
    # Core inference + execution helper (used by AUTO and SHARED states)
    # ------------------------------------------------------------------
    def _run_inference_and_execute(self, command_joint_velocity,
                                   gripper_command, blending: bool):
        """
        Runs one full RTC inference cycle.

        Parameters
        ----------
        command_joint_velocity : ndarray (H, 7)
            Joystick-derived velocity chunk.  Ignored when blending=False.
        gripper_command : float
            Target gripper position for this iteration.
        blending : bool
            True  → SHARED mode  (policy_blending applied)
            False → AUTO  mode   (pure VLA output)

        Returns
        -------
        action_new : ndarray or None
            The freshly inferred action chunk, or None on failure.
        observed_delay : int
        """
        with self.mutex:
            if self.A_cur is None:
                s_current = self.s_min
            else:
                s_current = self.H - len(self.pc._trajectory_queue) * self.action_quat

        if s_current < self.s_min:
            return None, 0  # not ready yet

        with self.mutex:
            s = int(s_current)
            if self.A_cur is not None:
                A_prev        = self.A_cur[s:self.H]
                d_estimated   = max(list(self.Q.queue))
            else:
                A_prev        = None
                d_estimated   = 0

        cur_q              = np.array(self.pc.get_current_joint_position())
        cur_gripper_position = self.gripper_width
        left_img           = self.left_img
        wrist_img          = self.wrist_img
        
        

        # Build proposed_action
        if blending and A_prev is not None:
            proposed_action = np.zeros((self.H, 32), dtype=np.float32)
            proposed_action[:len(A_prev), :8] = A_prev
            proposed_action[d_estimated:, :7] = command_joint_velocity[:self.H - d_estimated]
            proposed_action[d_estimated:, 7]  = gripper_command
        elif not blending and A_prev is not None:
            proposed_action = np.zeros((self.H, 32), dtype=np.float32)
            proposed_action[:len(A_prev), :8] = A_prev
            A_prev[:, 7] *= float(self.gripper_state == 'close')  # enforce current gripper state
        else:
            proposed_action = None

        queue_len_before = len(self.pc._trajectory_queue) * self.action_quat

        input_data = {
            "observation/exterior_image_1_left": left_img,
            "observation/wrist_image_left":      wrist_img,
            "observation/joint_position":        cur_q,
            "observation/gripper_position":      np.array([self.gripper_width], dtype=np.float32),
            "prompt":           self.current_prompt,
            "proposed_action":  proposed_action,
            "d":                d_estimated,
            "s":                s,
            # "s":                self.H,
        }
        self.client.update_data(input_data)

        action_new = self.client.send_request()
        if action_new is None:
            return None, 0

        # --- Consistency check: reject wildly divergent chunks ---
        if not self._is_action_consistent(action_new, observed_delay=d_estimated):
            print(
                '[ConsistencyCheck] action diverged from previous chunk — dropping, will re-infer')
            return None, 0

        with self.mutex:
            queue_len_after = len(self.pc._trajectory_queue) * self.action_quat
            observed_delay  = max(0, queue_len_before - queue_len_after)
            if self.Q.full():
                self.Q.get()
            self.Q.put(observed_delay)
            self.A_cur = action_new.copy()

        return action_new, observed_delay

    def _is_action_consistent(self, action_new: np.ndarray,
                               observed_delay: int,
                               pos_threshold: float = 0.15,
                               max_retries: int = 3) -> bool:
        """
        Compare the newly inferred action chunk with the tail of the previous
        chunk (A_cur) at the hand-off point.  Returns False when the jump is
        larger than `pos_threshold` (in joint-space L2 norm at the first
        overlapping step), signalling the caller to discard and re-infer.

        Parameters
        ----------
        action_new      : (H, ≥7) freshly inferred chunk (joint velocities)
        observed_delay  : actual delay steps measured during this inference
        pos_threshold   : maximum tolerated L2 norm of the integrated-position
                          difference at the hand-off step [rad]
        max_retries     : increment internal counter; caller should give up
                          after this many consecutive rejections to avoid
                          infinite loops (counter resets on acceptance).
        """
        with self.mutex:
            A_prev = self.A_cur  # snapshot under lock

        if A_prev is None:
            # No previous chunk — always accept the first one
            self._consistency_reject_count = 0
            return True

        # Guard: don't loop forever if the policy keeps diverging
        reject_count = getattr(self, '_consistency_reject_count', 0)
        if reject_count >= max_retries:
            self.get_logger().warn(
                f'[ConsistencyCheck] {reject_count} consecutive rejections — accepting anyway')
            self._consistency_reject_count = 0
            return True

        start_new  = min(observed_delay, self.H - 1)
        start_prev = min(observed_delay, self.H - 1)

        # Integrate one step from current joint position with each chunk
        cur_j = np.array(self.pc.get_current_joint_position())
        pos_new  = cur_j + action_new[start_new,  :7] * self.dt
        pos_prev = cur_j + A_prev[start_prev, :7] * self.dt

        diff = float(np.linalg.norm(pos_new - pos_prev))
        consistent = diff <= pos_threshold
        # print(f'[ConsistencyCheck] diff={diff:.4f} rad at hand-off point '
        #       f'(threshold={pos_threshold:.4f}, observed_delay={observed_delay}, ')

        if consistent:
            self._consistency_reject_count = 0
        else:
            self._consistency_reject_count = reject_count + 1
            print(
                f'[ConsistencyCheck] diff={diff:.4f} rad > threshold={pos_threshold:.4f}'
                f' (reject #{self._consistency_reject_count})')

        return consistent

    def _send_trajectory(self, action_new, observed_delay,
                         command_joint_velocity, blending: bool):
        """
        Integrate action chunk into joint positions and send to robot.
        """
        start_idx = min(observed_delay, self.H - 1)
        remaining_actions = action_new[start_idx:, :7]

        # max_acc = self.acceleration_calculation(action_new[start_idx:])
        # if max_acc > 300.0:
        #     remaining_actions = command_joint_velocity[:len(command_joint_velocity)- start_idx, :7]
        # elif blending:
        #     remaining_actions = self.policy_blending(
        #         remaining_actions, command_joint_velocity[:len(command_joint_velocity) - start_idx])

        if len(remaining_actions) == 0:
            return
        

        joint_positions = np.zeros((min(self.H, start_idx + len(remaining_actions)), 7))
        cur_j = np.array(self.pc.get_current_joint_position())
        if self.joint_positions_prev is not None:  
            idx = np.argmin(np.linalg.norm(self.joint_positions_prev- cur_j[None, :], axis=1))
            error = np.linalg.norm(cur_j-self.joint_positions_prev[idx])
            if error < 0.05:
                cur_j = self.joint_positions_prev[idx]

        joint_positions[start_idx] = cur_j + remaining_actions[0] * self.dt
        for i in range(1, len(remaining_actions)):
            joint_positions[start_idx + i] = joint_positions[start_idx + i-1] + remaining_actions[i] * self.dt
        
        if start_idx >= 1:
            # reverse integration
            for i in range(start_idx-1, -1, -1):
                joint_positions[i] = joint_positions[i+1] - self.A_cur[i][:7] * self.dt
                


        # joint_positions = np.zeros(
        #     (min(self.H, start_idx + len(remaining_actions)), 7))
        # cur_j = np.array(self.pc.get_current_joint_position())

        # joint_positions[start_idx] = cur_j + remaining_actions[0] * self.dt
        # for i in range(1, len(remaining_actions)):
        #     joint_positions[start_idx + i] = (
        #         joint_positions[start_idx + i - 1] + remaining_actions[i] * self.dt)

        # if start_idx >= 1:
        #     for i in range(start_idx - 1, -1, -1):
        #         joint_positions[i] = (
        #             joint_positions[i + 1] - action_new[i][:7] * self.dt)

        
                
        exe_joint_positions = joint_positions[start_idx+2:]
        if blending:
            exe_joint_positions = self.truncate_trajectory(exe_joint_positions, remaining_actions[2:], max_displacement=np.linalg.norm(command_joint_velocity[0, :7]* 0.6)) # 根据当前命令速度动态调整截断阈值


        # --- Joint limit clipping (80% of FR3 safety limits) ---
        _Q_MIN = np.array([-2.6437, -1.6837, -2.8007, -2.9421, -2.7065,  0.0825, -2.7973]) * 0.95
        _Q_MAX = np.array([ 2.6437,  1.6837,  2.8007, -0.2518,  2.7065,  3.6525,  2.7973]) * 0.95
        if len(exe_joint_positions) > 0:
            # violations = (exe_joint_positions < _Q_MIN) | (exe_joint_positions > _Q_MAX)
            # if violations.any():
            #     bad_joints = np.unique(np.where(violations)[1])
                # self.get_logger().warn(
                #     f'[JointLimit] clipping joints {bad_joints.tolist()} to 80% limit')
            exe_joint_positions = np.clip(exe_joint_positions, _Q_MIN, _Q_MAX)

        self.pc.visualize_trajectory(exe_joint_positions, self.pc.panda)
        self.pc.goto_joint_trajectory_pos(
            exe_joint_positions,
            alignment=False,
            non_blocking=True,
            replace_queue=True,
        )
        self.joint_positions_prev = exe_joint_positions

        cur_joint_position = np.array(self.pc.get_current_joint_position())
        self.last_joint_position = cur_joint_position
        if blending:
            self.last_execution_time = time.time()

    # ------------------------------------------------------------------
    # FSM state handlers
    # ------------------------------------------------------------------
    def _handle_stop(self, twist, voice):
        """
        STOP state: robot is fully stationary.
        Transition (1): prompt no longer contains 'stop' → AUTO
        """
        self.stop()

        # keep recording voice even in stop state
        self.voice_recorder.update_button(voice > 0.5)

        if not self._prompt_is_stop():
            self._transition(RobotState.AUTO)

    def _handle_auto(self, twist):
        """
        AUTO state: robot follows VLA autonomously.
        Transition (2): prompt contains 'stop'           → STOP
        Transition (3): VLA predicts gripper state change → MANUAL
        Transition (5): user joystick command detected    → SHARED
        Transition (Y): Y button pressed                  → MANUAL
        """
        vx, vy, vz, wx, wy, wz, home, grasp, release, auto, voice = twist

        # (2)
        if self._prompt_is_stop():
            self.stop()
            self._transition(RobotState.MANUAL)
            return

        # (5) user motion command (non-voice axes)
        if self._has_user_command(twist):
            self._transition(RobotState.SHARED)
            return

        if self.pc.robot_error or self.left_img is None or self.wrist_img is None:
            time.sleep(0.01)
            return

        while self.H - len(self.pc._trajectory_queue) * self.action_quat < self.s_min:
            time.sleep(0.01)
            return
        

        # Run inference (AUTO = no blending)
        dummy_velocity = np.zeros((self.H, 7), dtype=np.float32)

        action_new, observed_delay = self._run_inference_and_execute(
            dummy_velocity, None, blending=False)

        if action_new is None:
            time.sleep(0.01)
            return

        # (3) VLA wants gripper change → go to MANUAL, do NOT execute gripper
        start_idx = min(observed_delay, self.H - 1)
        
        if self._vla_wants_gripper_change(action_new, start_idx+1) and self.detector.results is not None and self.detector.results[0]['pred'] == 1:
            self._transition(RobotState.MANUAL)
            return



        self._send_trajectory(action_new, observed_delay, dummy_velocity, blending=False)
        cur_pose = self.pc.get_ee_pose()
        self.logger.info(f'[Logging] Current pose: {cur_pose.flatten().tolist()}')
        time.sleep(0.05)

    def _handle_shared(self, twist):
        """
        SHARED CONTROL state: user joystick input blended with VLA.
        Transition (2): prompt contains 'stop'        → STOP
        Transition (6): no user command this iteration → AUTO
        """
        vx, vy, vz, wx, wy, wz, home, grasp, release, auto, voice = twist

        # (2)
        if self._prompt_is_stop():
            self.stop()
            self._transition(RobotState.MANUAL)
            return

        # (6) user released all controls — wait for cooldown before returning to AUTO
        if not self._has_user_command(twist):
            elapsed = time.time() - self._last_shared_enter_time
            if elapsed >= self._shared_cooldown_sec:
                self._transition(RobotState.AUTO)
                print(f"FSM: SHARED → AUTO (cooldown {elapsed:.1f}s elapsed)")
            else:
                print(
                    f"FSM: AUTO blocked from SHARED (cooldown {self._shared_cooldown_sec - elapsed:.1f}s remaining)")
                self.stop()
                time.sleep(0.01)
            return

        if self.pc.robot_error or self.left_img is None or self.wrist_img is None:
            time.sleep(0.01)
            return

        self._last_shared_enter_time = time.time()
        cur_q = np.array(self.pc.get_current_joint_position())
        pos_command = np.array([vx, vy, vz]) # + np.random.normal(0, 0.03, size=3) 
        ang_command = np.array([wx, wy, wz]) # + + np.random.normal(0, 0.03, size=3) 

        command_joint_velocity = velocity_based_control(
            self.pc.panda, cur_q, pos_command, ang_command, onbase=True, T_tcp_robotiq=self.pc.T_tcp_robotiq.as_matrix())
        command_joint_velocity = np.repeat(
            command_joint_velocity.reshape(1, -1), self.H, axis=0) * 2

        decaying_factors = np.array([0.8**t for t in range(self.H)])[:, None]
        command_joint_velocity = command_joint_velocity * decaying_factors

        # Gripper from joystick
        if self.gripper_state == 'open' and grasp > 0.5:
            gripper_command = 0.8
        elif self.gripper_state == 'close' and release > 0.5:
            gripper_command = 0.0
        else:
            gripper_command = float(self.gripper_state == 'close')

        action_new, observed_delay = self._run_inference_and_execute(
            command_joint_velocity, gripper_command, blending=True)

        if action_new is None:
            time.sleep(0.01)
            return

        # Execute gripper command from joystick
        target_gripper, gripper_flag = self.grasp(gripper_command)
        if gripper_flag:
            self.pc.grasp(target_gripper)
            return


        self._send_trajectory(action_new, observed_delay,
                               command_joint_velocity, blending=True)
        self.total_commands += 1
        cur_pose = self.pc.get_ee_pose()
        self.logger.info(f'[Logging] Current pose: {cur_pose.flatten().tolist()}')
        self.logger.info(f'[Logging] cur_iter: {self.total_commands}')
        self.logger.info(f'[Logging] Current command: {twist.tolist()}')
        # self.print_detector_logits(np.abs(float(self.gripper_state == 'close') - action_new[observed_delay+1, 7]))
        # time.sleep(0.03)

    def _handle_intervention(self, twist):

        # if self._prompt_is_stop():
        #     self.stop()
        #     self._transition(RobotState.MANUAL)
        #     return
        if self.last_execution_time is not None and (time.time() - self.last_execution_time) > 0.01:
            self.stop()
            time.sleep(0.01)
            return

        t1 = time.time()
        # (6) user released all controls
        if not self._has_user_command(twist):
            time.sleep(0.01)
            return

        # self._handle_intervention_manual(twist)
        # return
        if self.current_prompt is not None and ('stop' in self.current_prompt or 'Stop' in self.current_prompt):
            self._handle_intervention_manual(twist)
            return
        
        vx, vy, vz, wx, wy, wz, home, grasp, release, auto, voice = twist
        if self.pc.robot_error or self.left_img is None or self.wrist_img is None:
            time.sleep(0.01)
            return
        
        with self.mutex:
            if self.A_cur is None:
                # 如果系统刚启动或被重置，强制进度达标以立即触发推理
                s_current = self.s_min 
            else:
                # 💡 核心改进：H 减去底层队列中还未执行的动作数，就是当前 Chunk 已经执行了的动作数
                s_current = self.H - len(self.pc._trajectory_queue)

        while s_current < self.s_min:
            time.sleep(0.01) # 还没执行够 s_min 步，继续等
            with self.mutex:
                s_current = self.H - len(self.pc._trajectory_queue)
                
        t2 = time.time()
        

        cur_q = np.array(self.pc.get_current_joint_position())
        pos_command = np.array([vx, vy, vz]) # + np.random.normal(0, 0.03, size=3) 
        ang_command = np.array([wx, wy, wz]) # + + np.random.normal(0, 0.03, size=3) 




        command_joint_velocity = velocity_based_control(
            self.pc.panda, cur_q, pos_command, ang_command, onbase=True, T_tcp_robotiq=self.pc.T_tcp_robotiq.as_matrix())
        command_joint_velocity = np.repeat(
            command_joint_velocity.reshape(1, -1), self.H, axis=0) * 2

        decaying_factors = np.array([0.8**t for t in range(self.H)])[:, None]
        command_joint_velocity = command_joint_velocity * decaying_factors
        

        target_gripper = float(self.gripper_state == 'close')
        if self.gripper_state == 'open' and grasp > 0.5:
            target_gripper, gripper_flag = self.grasp(0.8)
            self.pc.grasp(target_gripper)
            self._transition(RobotState.AUTO)
            return
        if self.gripper_state == 'close' and release > 0.5:
            target_gripper, gripper_flag = self.grasp(0.0)
            self.pc.grasp(target_gripper)
            # self._transition(RobotState.AUTO)
            return

        t3 = time.time()

        action_new, observed_delay = self._run_inference_and_execute(
            command_joint_velocity, target_gripper, blending=True)

        if action_new is None:
            time.sleep(0.01)
            return

        t4 = time.time()
        self._send_trajectory(action_new, observed_delay,
                               command_joint_velocity, blending=True)
        t5 = time.time()
        self.total_commands += 1
        cur_pose = self.pc.get_ee_pose()
        self.logger.info(f'[Logging] Current pose: {cur_pose.flatten().tolist()}')
        self.logger.info(f'[Logging] cur_iter: {self.total_commands}')
        self.logger.info(f'[Logging] Current command: {twist.tolist()}')
        # print(f'[Timing] inference wait: {t2 - t1:.2f}s, inference: {t4 - t3:.2f}s, send traj: {t5 - t4:.2f}s')


    def _handle_intervention_manual(self, twist):
        """
        MANUAL state: pure user teleoperation, no VLA.
        Transition (2): prompt contains 'stop'  → STOP
        Transition (4): user changes gripper     → AUTO
        Transition (Y): Y button pressed         → AUTO  (handled in inference_loop)
        """
        vx, vy, vz, wx, wy, wz, home, grasp, release, auto, voice = twist


        if self.pc.robot_error:
            time.sleep(0.1)
            return

        cur_q = np.array(self.pc.get_current_joint_position())
        pos_command = np.array([vx, vy, vz])
        ang_command = np.array([wx, wy, wz])



        # Execute pure user velocity
        cur_q = np.array(self.pc.get_current_joint_position())
        command_joint_velocity = velocity_based_control(self.pc.panda, cur_q, pos_command, ang_command, onbase=True, T_tcp_robotiq=self.pc.T_tcp_robotiq.as_matrix())
        desired_joint_position = cur_q + command_joint_velocity * 0.5
        self.pc.react_control_flag = True
        self.pc.joint_command_msg.name = self.pc.joint_names
        self.pc.joint_command_msg.position = desired_joint_position.tolist()

        # (4) user changed gripper state → return to AUTO
        if self.gripper_state == 'open' and grasp > 0.5:
            target_gripper, gripper_flag = self.grasp(0.8)
            self.pc.grasp(target_gripper)
            self._transition(RobotState.AUTO)
            return
        if self.gripper_state == 'close' and release > 0.5:
            target_gripper, gripper_flag = self.grasp(0.0)
            self.pc.grasp(target_gripper)
            # self._transition(RobotState.AUTO)
            return

        # (7) accumulated user commands exceed threshold → revert to SHARED
        # if self._has_user_command(twist):
        #     self._intervention_user_cmd_count += 1
        #     if self._intervention_user_cmd_count >= self._intervention_timeout_iters:
        #         self.get_logger().info(
        #             f"User commanded {self._intervention_user_cmd_count} iterations in MANUAL "
        #             f"— reverting MANUAL → SHARED"
        #         )
        #         self._transition(RobotState.SHARED)
        #         return

        self.total_commands += 1
        cur_pose = self.pc.get_ee_pose()
        self.logger.info(f'[Logging] Current pose: {cur_pose.flatten().tolist()}')
        self.logger.info(f'[Logging] cur_iter: {self.total_commands}')
        self.logger.info(f'[Logging] Current command: {twist.tolist()}')
        # self.print_detector_logits(1.0)
        time.sleep(0.05)

    # ------------------------------------------------------------------
    # Main FSM loop
    # ------------------------------------------------------------------
    def inference_loop(self):
        self.get_logger().info("FSM control loop started.")
        # self.refine_flag = input("whether refine the language instruction? (y/n): ").strip().lower() == 'y'
        self.refine_flag = False
        # self.get_logger().info(f"Refine language instruction: {self.refine_flag}")
        self.voice_recording_flag = False
        start_teleop = False
        start_time = None
        
        self.logger.info(f'[Logging] Current mode: {self.fsm_state.value.upper()}')
        while rclpy.ok():
            # --- read joystick ---
            twist = self.joy_listener.get_twist_array()
            if twist is None:
                time.sleep(0.01)
                continue
            loop_start_time = time.time()

            # --- controller watchdog ---
            if self._recovering:
                time.sleep(0.1)
                continue
            if (self.fsm_state != RobotState.STOP and
                    time.time() - self._last_js_stamp > self._js_stale_sec):
                self.get_logger().error(
                    f"No /joint_states for {self._js_stale_sec}s — controller crash detected.")
                self.fsm_state = RobotState.STOP   # direct: bypass _transition / no rumble
                self.A_cur = None
                self._recovering = True
                threading.Thread(target=self._do_recovery, daemon=True).start()
                time.sleep(0.1)
                continue

            if not start_teleop and np.linalg.norm(twist) > 0.001:
                start_teleop = True
                self.logger.info("[Logging] Starting teleoperation...")
                start_time = time.time()

            vx, vy, vz, wx, wy, wz, home, grasp, release, auto, voice = twist

            # voice recording handled here regardless of state
            self.voice_recorder.update_button(voice > 0.5)
            if voice > 0.5:
                # While recording, freeze motion
                if self.voice_recording_flag is False:
                    self.logger.info("[Logging] Voice recording started.")
                    self.voice_recording_flag = True

                time.sleep(0.01)
                self.stop()
                continue

            
            if self.current_prompt != self.voice_recorder.prompt and 'stop' not in self.voice_recorder.prompt and 'Stop' not in self.voice_recorder.prompt:
                self.current_prompt = self.voice_recorder.prompt
                # self._transition(RobotState.AUTO)
                self.logger.info("[Logging] Voice recording stopped.")
                self.voice_recording_flag = False
                self.logger.info(f'[Logging] Current prompt: {self.current_prompt}')
            
            
            # --- Y button (auto): toggle AUTO ↔ MANUAL ---
            auto_btn = auto > 0.5
            if auto_btn and not self._auto_btn_prev:
                if self.fsm_state == RobotState.MANUAL:
                    self.A_cur = None  # restart inference fresh
                    self._transition(RobotState.AUTO)
                elif self.fsm_state in (RobotState.AUTO, RobotState.SHARED):
                    self._transition(RobotState.MANUAL)
            self._auto_btn_prev = auto_btn


            if twist[-5] == 1 or (start_time is not None and time.time() - start_time > 7 * 60):  # either "home" button pressed or 7 minutes elapsed
                if start_teleop:
                    start_teleop = False
                    self.logger.info("[Logging] Task Finished.")
                    if start_time is not None and time.time() - start_time > 7 * 60:
                        self.logger.info("[Logging] Time limit reached (7 minutes).")
                    result = input("Please enter the progress: ")
                    self.logger.info(f"[Logging] progress: {result}")
                with self.mutex:
                    self.pc.home()
                    self.stop()
                    self.voice_recorder.update_prompt("stop")
                    self.current_prompt = "stop"
                    continue
            
            if self.wrist_img_raw is not None:
                self.detector.predict_batch_async([self.wrist_img_raw], gripper_closed=(self.gripper_state=='close'))

            # Record detector result + gripper state into history
            if self.detector.results is not None:
                self._detector_history.append({
                    'time': time.time(),
                    'pred': self.detector.results[0]['pred'],
                    'prob_1': self.detector.results[0]['prob_1'],
                    'gripper_state': self.gripper_state,
                })
            # print("preprocess time: {:.3f}s".format(time.time() - loop_start_time))

            # --- FSM dispatch ---
            if self.fsm_state == RobotState.STOP:
                self._handle_stop(twist, voice)

            elif self.fsm_state == RobotState.AUTO:
                self._handle_auto(twist)

            elif self.fsm_state == RobotState.SHARED:
                self._handle_shared(twist)

            elif self.fsm_state == RobotState.MANUAL:
                self._handle_intervention(twist)
            print(f"[{self.fsm_state.value}] iteration: {time.time() - loop_start_time:.3f}s")

            # print(f"[{self.fsm_state.value}] iteration: {time.time() - start_time:.3f}s")


# =============================================================================
# Entry point
# =============================================================================
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

    os.makedirs("logs/{}".format(user_name), exist_ok=True)
    method_idx = 3
    method = methods_dict[method_idx]
    log_file_path = os.path.expanduser(os.path.join("logs/{}".format(user_name), '_'.join((current_time,user_name,method))+".log"))
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    node = Assistron(logger)

    executor = MultiThreadedExecutor(12)
    executor.add_node(node)
    executor.add_node(node.pc)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    node.pc.release()
    node.pc.home()
    

    try:
        node.inference_loop()
    except KeyboardInterrupt:
        node.get_logger().info("程序被用户手动终止")
    finally:
        node.voice_recorder.close()
        node.stop()
        if USE_PYREALSENSE:
            node._rs_stop_event.set()
            node._rs_thread.join(timeout=2.0)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
