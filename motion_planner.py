# Third Party
import torch
import time
import sys
import zmq
import pickle
import numpy as np

# CuRobo Imports
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
# --- 新增 Mesh ---
from curobo.geom.types import WorldConfig, Mesh, Cuboid

class MotionPlanner:
    def __init__(self, robot_yml="ur5e.yml", debug=True):
        self.debug = debug
        
        # 默认占位符
        world_config_placeholder = {
            "cuboid": {
                "placeholder": {
                    "dims": [0.1, 0.1, 0.1],
                    "pose": [0.0, 0.0, -10.0, 1, 0, 0, 0.0], # 放远点
                },
            },
        }
        
        t = time.time()
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_yml,
            world_config_placeholder,
            collision_cache={"obb": 10, "mesh": 10},
            interpolation_dt=0.05,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        if self.debug:
            print(f"[MotionServer] Planner loaded in {time.time()-t:.3f}s")

    def plan(self, start_joint_state, target_pose, plan_config=None):
        # (保持原有的 plan 逻辑不变)
        goal_pose = Pose.from_list(target_pose)  
        start_joint_tensor = torch.tensor(start_joint_state, dtype=torch.float32).reshape(1, -1).cuda()
        start_state = JointState.from_position(start_joint_tensor)
        
        if plan_config is None:
            plan_config = MotionGenPlanConfig(max_attempts=60) 

        result = self.motion_gen.plan_single(start_state, goal_pose, plan_config)        
        
        success = result.success.detach().cpu().item()
        waypoints_np = None
        if success:
            traj = result.get_interpolated_plan()
            waypoints_np = traj.position.detach().cpu().numpy()
            
        return waypoints_np, success

    def update_world(self, cuboids_dict=None, pcl_array=None):
        """
        根据传入的 cuboid 字典和 pcl numpy 数组构建 WorldConfig
        """
        t = time.time()
        self.motion_gen.clear_world_cache()
        
        world_cfg_args = {}

        # 1. 处理 Cuboids
        if cuboids_dict is not None and len(cuboids_dict) > 0:
            # 如果传入的是 dict 格式的 cuboid (例如: {'table': {'dims':..., 'pose':...}})
            # CuRobo 的 WorldConfig 可以直接接受 dict 转换
            # 或者我们显式构建 Cuboid 对象，这里直接用 WorldConfig 的灵活性
            world_cfg_args['cuboid'] = WorldConfig.from_dict({'cuboid': cuboids_dict}).cuboid

        # 2. 处理 PointCloud -> Mesh
        if pcl_array is not None and len(pcl_array) > 0:
            try:
                # 确保是 tensor 且在 cuda 上 (CuRobo 偏好)
                # Mesh.from_pointcloud 需要 numpy 或 tensor
                # 注意：pcl_array 应该是 (N, 3) 的 numpy 数组
                
                # 创建 Mesh 对象
                # pose 为 mesh 的原点，通常点云是在世界系下的，所以 pose 是原点
                mesh_obstacle = Mesh.from_pointcloud(
                    pcl_array, 
                    pose=[0,0,0,1,0,0,0], 
                    name="scene_pcl"
                )
                world_cfg_args['mesh'] = [mesh_obstacle]
                
            except Exception as e:
                print(f"[MotionServer] Error creating mesh from PCL: {e}")

        # 3. 更新 CuRobo
        if len(world_cfg_args) > 0:
            try:
                # WorldConfig 既可以接收 mesh=[Obj], cuboid={dict}
                world_config = WorldConfig(**world_cfg_args)
                self.motion_gen.update_world(world_config)
                if self.debug:
                    pcl_size = len(pcl_array) if pcl_array is not None else 0
                    print(f"[MotionServer] World updated: {len(world_cfg_args)} types | PCL: {pcl_size} pts | Time: {time.time()-t:.3f}s")
                return True
            except Exception as e:
                print(f"[MotionServer] Update World Error: {e}")
                return False
        
        return True # 空更新也算成功

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5556")
    print("[MotionServer] ZMQ REP Server bound to tcp://*:5556")

    mp = MotionPlanner(robot_yml="franka.yml", debug=True)
    print("[MotionServer] Ready.")

    while True:
        try:
            msg = socket.recv()
            request = pickle.loads(msg)
            
            cmd = request.get('cmd')
            response = {}

            if cmd == 'plan':
                start = request['start']
                target = request['target']
                waypoints, success = mp.plan(start, target)
                response = {'success': success, 'waypoints': waypoints}

            elif cmd == 'update_world':
                # 解析参数: cuboids (dict) 和 pcl (numpy array)
                cuboids = request.get('cuboids', None)
                pcl = request.get('pcl', None)
                
                ok = mp.update_world(cuboids_dict=cuboids, pcl_array=pcl)
                response = {'success': ok}

            elif cmd == 'ping':
                response = {'status': 'ok'}

            socket.send(pickle.dumps(response))

        except Exception as e:
            print(f"[MotionServer] Error: {e}")
            try:
                socket.send(pickle.dumps({'success': False, 'error': str(e)}))
            except:
                pass

if __name__ == "__main__":
    run_server()