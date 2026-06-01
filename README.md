# Assistron: Bayesian Shared Autonomy with Off-the-shelf Vision-Language-Action Models

<!-- > **CoRL 2026 Submission** — Shared autonomy framework for assistive manipulation, no VLA fine-tuning required. -->


## 🚀 Highlights

- **Zero fine-tuning**: Assistron operates on a frozen π0.5 VLA policy, preserving its open-world generalization without catastrophic forgetting.
- **Phase-aware intervention**: A lightweight ResNet-18 interaction detector automatically triggers human assistance only at contact-rich bottlenecks (grasping, insertion, release), minimizing operator burden.
- **Bayesian policy blending**: Human joystick commands are fused into the VLA's flow-matching denoising process as an analytical posterior guidance term — outperforming naive linear blending by 4.1× in smoothness.
- **91.3% task success** on a 5-subtask scene recovery benchmark (vs. 13.7% for VLA-only), while cutting active user control time in half compared to direct teleoperation.
- **Voice + joystick interface**: Verbal commands (via Whisper) drive macro-reaching; low-bandwidth joystick corrections handle fine interaction.

---

## 💡 Key Insight

Real-world VLA failures are rarely *semantic* — they are *spatial*, concentrated in the short temporal window before contact (grasp, insertion, release). Assistron exploits this asymmetry:

```
   Macro-reaching          Contact-rich phase
 ┌──────────────────┐    ┌─────────────────────────┐
 │  Frozen VLA      │───▶│  Human + VLA posterior  │
 │  (full authority)│    │  blending (shared ctrl) │
 └──────────────────┘    └─────────────────────────┘
         Auto mode               Assist mode
```

The system policy is:

$$\pi_{\text{sys}}(\boldsymbol{a}|\boldsymbol{s}) = (1 - \mathbb{I}_{\text{int}})\,\pi_{\text{vla}}(\boldsymbol{a}|\boldsymbol{s}) + \mathbb{I}_{\text{int}}\,\pi_{\text{shared}}(\boldsymbol{a}|\boldsymbol{s}, \boldsymbol{u})$$

where the intervention indicator $\mathbb{I}_{\text{int}}$ is triggered by the interaction detector or by a non-zero joystick input. The shared policy adds an analytical flow-matching guidance term:

$$\hat{v}(\boldsymbol{a}_t, \boldsymbol{u}) = \hat{v}(\boldsymbol{a}_t) + \left(\frac{1-t}{t}\right)(\boldsymbol{u} - \hat{\boldsymbol{a}}_1)^T\left(\frac{(1-t)^2}{(1-t)^2 + t^2}\boldsymbol{I} + \boldsymbol{\Sigma}_u\right)^{-1}$$

---

## 🤖 Hardware Setup

| Component | Model |
|---|---|
| Robot arm | Franka Research 3 |
| Gripper | Robotiq 2F-85 |
| Exterior camera | Intel RealSense D435i |
| Wrist camera | Intel RealSense D456 |
| User interface | Xbox joystick + microphone |

---

## 🛠️ Installation

### 1. Clone and install dependencies

```bash
git clone <this-repo>
cd assistron
pip install -r requirements.txt
```

### 2. FR3 control interface (Franka controller)

Assistron drives the Franka Research 3 via the **`joint_position_impedance_controller`** provided by the [fr3_control_interface](https://github.com/mousecpn/fr3_control_interface) repository. Clone and build it inside your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/mousecpn/fr3_control_interface.git
cd ~/ros2_ws
colcon build --packages-select franka_control_wrappers
source install/setup.bash
```

The controller publishes and subscribes on the topic:
```
/joint_position_impedance_controller/joint_states
```
which is the `control_topic` used by `panda_control.py`. Make sure this controller is loaded and active before launching Assistron.

> **Dependencies of fr3_control_interface**: ROS 2 Humble, [roboticstoolbox-python](https://github.com/petercorke/robotics-toolbox-python), [curobo](https://github.com/NVlabs/curobo), [sdfsc](https://github.com/lichadelz/sdfsc).

### 3. ROS 2 dependencies

The main node requires ROS 2 (tested on Humble) with the following packages:

```bash
sudo apt install ros-humble-franka-msgs ros-humble-cv-bridge
pip install openpi-client  # physical intelligence client
```

### 4. Whisper (speech recognition)

```bash
pip install faster-whisper soundfile librosa
```

### 5. π0.5 Triton inference

```bash
pip install torch triton zmq einops transformers
```

#### Convert JAX checkpoint to Triton format

Download the π0.5 DROID checkpoint from the [OpenPI](https://github.com/Physical-Intelligence/openpi) repository, then convert:

```bash
python3 pi05_triton/convert_from_jax_pi05.py \
    --jax_path /path/to/openpi/checkpoints/pi05_droid \
    --output /path/to/pi05_droid_triton.pkl \
    --prompt "do something" \
    --tokenizer_path google/paligemma-3b-pt-224
```

> **Note**: `--tokenizer_path` accepts either a local directory or a Hugging Face model ID. The `paligemma-3b-pt-224` tokenizer can be downloaded from Hugging Face:
> ```bash
> huggingface-cli download google/paligemma-3b-pt-224 --local-dir /path/to/paligemma-3b-pt-224
> ```

### 6. Edit server paths

Open `pi05_triton/pi05_zmq_server.py` and update the hard-coded paths near the `main()` function:

```python
triton_path    = "/path/to/pi05_droid_triton.pkl"
jax_path       = "/path/to/openpi/checkpoints/pi05_droid"
norm_stats_dir = "/path/to/openpi/checkpoints/pi05_droid/assets/droid"
tokenizer_path = "google/paligemma-3b-pt-224"   # or local path
```

---

## ▶️ Deployment

Assistron requires **three terminal windows** running concurrently.

### Terminal 1 — Whisper HTTP server

Starts the speech recognition service on port `43100` (default).

```bash
cd assistron/whisper_streaming

python3 whisper_http_server.py \
    --host 0.0.0.0 \
    --port 43100 \
    --backend faster-whisper \
    --model large-v3 \
    --lan en
```

Wait until the server prints `Listening on 0.0.0.0:43100` before proceeding.

> **Tip**: Add `--warmup-file /path/to/short.wav` to pre-warm the ASR model and reduce first-query latency.

---

### Terminal 2 — π0.5 ZMQ inference server

Loads the Triton-accelerated π0.5 model and serves inference requests over ZMQ on port `5555`.

```bash
cd assistron/pi05_triton

python3 pi05_zmq_server.py
```

The server prints `[*] Listening on tcp://*:5555` when ready. Inference runs on CUDA; ensure a GPU is available.

> **Tip**: Verify the server is working correctly with the included test client:
> ```bash
> python3 pi05_zmq_client.py
> ```

---

### Terminal 3 — Assistron main node

After both servers are running, launch the main ROS 2 control node:

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

cd assistron
python3 assistron.py
```

The node will:
1. Home the robot arm.
2. Enter **MANUAL** mode — full joystick teleoperation.
3. Press **Y** on the Xbox controller to switch to **AUTO** mode (VLA-driven).
4. The system automatically transitions to **ASSIST** mode when the interaction detector fires or the user moves the joystick.

---

## 🎮 Joystick Controls

| Input | Action |
|---|---|
| Left stick + trigger/bumper | End-effector translation |
| Right stick + trigger/bumper | End-effector rotation |
| **A** button | Close gripper |
| **B** button | Open gripper |
| **Y** button | Toggle AUTO / MANUAL mode |
| Hold **X** | Record verbal command (Whisper) |

<!-- ---

## 📁 Project Structure

```
assistron/
├── assistron.py                  # Main ROS 2 node (FSM + control loop)
├── panda_control.py              # Franka arm low-level interface
├── pi05_client.py                # ZMQ client wrapper for the VLA server
├── motion_planner.py             # Trajectory planning utilities
├── fr3_launch.py                 # Robot bringup helpers
├── interaction_detection/
│   ├── infer_dual_model.py       # Dual-model interaction detector (inference)
│   ├── success_model.py          # ResNet-18 classifier definition
│   ├── best_close_0.85.pt        # Weights: close-action detector
│   └── best_open_0.83.pt         # Weights: open-action detector
├── pi05_triton/
│   ├── pi05_infer.py             # Triton-accelerated π0.5 inference kernel
│   ├── pi05_zmq_server.py        # ZMQ server wrapping the VLA
│   ├── pi05_zmq_client.py        # Test/debug client
│   └── convert_from_jax_pi05.py  # JAX → Triton checkpoint converter
├── whisper_streaming/
│   ├── whisper_http_server.py    # Whisper HTTP transcription server
│   └── whisper_online.py         # Streaming ASR utilities
└── utils/
    ├── joy_listener.py           # Xbox joystick input parser
    ├── control.py                # Velocity-based control utilities
    ├── voice_transcript_client.py # HTTP client for Whisper server
    └── shared_control_status_windows.py  # Real-time status visualisation
``` -->

<!-- ---

## 📄 Citation

```bibtex
@inproceedings{assistron2026corl,
  title     = {Assistron: Bayesian Shared Autonomy with Off-the-shelf Vision-Language-Action Models},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
}
``` -->
