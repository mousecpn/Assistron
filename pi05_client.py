import pickle
import zmq
import time
import numpy as np
# =============================================================================
# ZMQ inference client (unchanged)
# =============================================================================
class pi05_client:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        server_address = "tcp://172.16.0.30:5555"
        print(f"正在连接到服务器 {server_address}...")
        self.socket.connect(server_address)
        self.data_buffer = {
            "observation/exterior_image_1_left": np.random.rand(224, 224, 3).astype(np.float32),
            "observation/wrist_image_left":      np.random.rand(224, 224, 3).astype(np.float32),
            "observation/joint_position":        np.random.randn(7).astype(np.float32),
            "observation/gripper_position":      np.random.randn(1).astype(np.float32),
            "prompt": "pick up the green block"
        }

    def update_data(self, new_data):
        self.data_buffer.update(new_data)

    def send_request(self):
        try:
            start_time = time.time()
            serialized_data = pickle.dumps(self.data_buffer)
            self.socket.send(serialized_data)
            reply_message = self.socket.recv()
            result = pickle.loads(reply_message)
            rtt = time.time() - start_time
            if result.get("status") == "success":
                return result["action"]
            else:
                print(f"❌ 服务器返回错误:\n{result.get('message')}")
                return None
        except zmq.ZMQError as e:
            print(f"ZMQ 错误: {e}")
            return None
