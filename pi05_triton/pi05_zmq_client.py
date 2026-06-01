import zmq
import pickle
import numpy as np
import time

def main():
    # 1. 设置 ZMQ Context 和 Socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ) # REQ (Request) socket 用于发送请求
    
    # 假设服务器运行在本地，端口为 5555
    server_address = "tcp://localhost:5555" 
    print(f"正在连接到服务器 {server_address}...")
    socket.connect(server_address)

    # 2. 构造符合输入要求的模拟数据 (Dummy data)
    # 在实际应用中，这里替换为您从真实相机和机器人本体读取的数据
    print("正在生成模拟输入数据...")
    propsoed_action =  np.random.randn(15, 32).astype(np.float32)
    left = np.random.rand(224, 224, 3).astype(np.float32)
    wrist = np.random.rand(224, 224, 3).astype(np.float32)
    joints = np.random.randn(7).astype(np.float32)
    gripper = np.random.randn(1).astype(np.float32)
    for i in range(5):
        inputs = {
            # 假设图像为 224x224 的 RGB 图像
            "observation/exterior_image_1_left": left,
            "observation/wrist_image_left": wrist,
            # 假设有 7 个维度的关节位置
            "observation/joint_position": joints,
            # 假设有 1 个维度的夹爪位置
            "observation/gripper_position": gripper,
            # 文本 prompt
            "prompt": "pick up the green block",
            "proposed_action": propsoed_action,
            "d": 1,
            "s": 8
        }

        # 3. 序列化并发送请求
        print("正在向服务器发送请求...")
        start_time = time.time()
        
        # 使用 pickle 序列化字典数据
        serialized_data = pickle.dumps(inputs)
        socket.send(serialized_data)

        # 4. 等待并接收服务器回复
        print("正在等待服务器响应...")
        reply_message = socket.recv()
        
        # 反序列化回复数据
        result = pickle.loads(reply_message)
        
        rtt = time.time() - start_time
        

        # 5. 处理结果
        if result.get("status") == "success":
            action = result["action"]
            print(f"✅ 推理成功！往返耗时 (RTT): {rtt:.3f} 秒")
            print(f"接收到的 action 形状: {action.shape}")
            print(f"第一步 action 示例: {action[0]}")
        else:
            print(f"❌ 服务器返回错误:\n{result.get('message')}")

if __name__ == "__main__":
    main()