import os
import json
import pickle
import sys
import traceback
import time

import numpy as np
import torch
import zmq
import einops
from PIL import Image
from pi05_infer import Pi05Inference


class Pi05Evaluator:
    """Minimal pi05+triton evaluator for DROID ZMQ inference."""

    def __init__(
        self,
        triton_path: str,
        norm_stats_dir: str,
        tokenizer_path: str,
        action_horizon: int = 15,
        action_dim: int = 8,
        noise_step: int = 5,
        discrete_state_input: bool = True,
        prompt: str = None,
    ):
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        self.norm_stats = self._load_norm_stats(norm_stats_dir)
        q01 = np.array(self.norm_stats["actions"]["q01"])
        q99 = np.array(self.norm_stats["actions"]["q99"])
        self._actions_q01 = q01
        self._actions_q99 = q99

        self._digitize_bins = np.linspace(-1, 1, 257)[:-1]

        with open(triton_path, "rb") as f:
            weights = pickle.load(f)

        self.policy = Pi05Inference(
            checkpoint=weights,
            num_views=2,
            chunk_size=action_horizon,
            tokenizer_path=tokenizer_path,
            max_tokenize_len=200,
            max_prompt_text=prompt,
            discrete_state_input=discrete_state_input,
            state_dim_for_max_prompt=8,
            noise_step=noise_step,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_dim = 32
        q01_t = torch.from_numpy(self._pad_to_dim(q01, target_dim)).to(dtype=torch.float32, device=device)
        q99_t = torch.from_numpy(self._pad_to_dim(q99, target_dim)).to(dtype=torch.float32, device=device)
        self.policy.buffers["actions_q01"].copy_(q01_t)
        self.policy.buffers["actions_q99"].copy_(q99_t)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_norm_stats(self, norm_stats_dir: str) -> dict:
        path = os.path.join(norm_stats_dir, "norm_stats.json")
        with open(path, "r") as f:
            return json.load(f)["norm_stats"]

    def _pad_to_dim(self, x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
        current = x.shape[axis]
        if current < target_dim:
            pad = [(0, 0)] * x.ndim
            pad[axis] = (0, target_dim - current)
            return np.pad(x, pad)
        return x

    def _parse_image(self, image) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
        return image

    def _resize_with_pad(self, image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
        pil = Image.fromarray(image)
        w, h = pil.size
        if w == width and h == height:
            return image
        ratio = max(w / width, h / height)
        new_h, new_w = int(h / ratio), int(w / ratio)
        resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)
        canvas = Image.new(resized.mode, (width, height), 0)
        canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
        return np.array(canvas)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0 * 2.0 - 1.0

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        dim = state.shape[-1]
        q01 = np.array(self.norm_stats["state"]["q01"])[:dim]
        q99 = np.array(self.norm_stats["state"]["q99"])[:dim]
        return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def _digitize_state(self, state_normed: np.ndarray) -> np.ndarray:
        return np.digitize(state_normed, bins=self._digitize_bins) - 1

    def normalize_actions(self, actions: np.ndarray, target_dim: int = 32) -> np.ndarray:
        q01 = self._pad_to_dim(self._actions_q01, target_dim)
        q99 = self._pad_to_dim(self._actions_q99, target_dim)
        return (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def unnormalize_actions(self, actions: np.ndarray, target_dim: int = 32) -> np.ndarray:
        q01 = self._pad_to_dim(self._actions_q01, target_dim)
        q99 = self._pad_to_dim(self._actions_q99, target_dim)
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    def _preprocess_inputs(self, data: dict) -> dict:
        """Build the model's input dict from raw observation data."""
        joint_pos = np.asarray(data["observation/joint_position"])
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([joint_pos, gripper_pos])
        state = self._normalize_state(state)
        state_tokens = self._digitize_state(state)

        base_image = self._normalize_image(self._resize_with_pad(self._parse_image(data["observation/exterior_image_1_left"])))
        wrist_image = self._normalize_image(self._resize_with_pad(self._parse_image(data["observation/wrist_image_left"])))

        return {
            "state_tokens": state_tokens,
            "images": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
            },
        }

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def infer(
        self,
        inputs: dict,
        copilot_noise: np.ndarray,
        initial_noise: np.ndarray,
        velocity_command: np.ndarray | None,
        rtc: bool = False,
        hajl: bool = False,
        d: int = 3,
        s: int = 8,
    ) -> np.ndarray:
        """Run one inference step. Returns actions array of shape (action_horizon, action_dim)."""
        preprocessed = self._preprocess_inputs(inputs)

        observation_images = torch.stack([
            torch.from_numpy(preprocessed["images"]["base_0_rgb"]),
            torch.from_numpy(preprocessed["images"]["left_wrist_0_rgb"]),
        ]).to(torch.float32).cuda(non_blocking=True)

        diffusion_noise = torch.from_numpy(
            np.random.randn(self.action_horizon, 32).astype(np.float32)
        ).cuda(non_blocking=True)

        copilot_noise_t = torch.from_numpy(copilot_noise).to(torch.float32).cuda(non_blocking=True)
        initial_noise_t = torch.from_numpy(initial_noise).to(torch.float32).cuda(non_blocking=True)
        velocity_t = (
            torch.from_numpy(velocity_command).to(torch.float32).cuda(non_blocking=True)
            if velocity_command is not None else None
        )

        self.policy.rtc = rtc
        self.policy.d = d
        self.policy.s = s
        self.policy.hajl = hajl if not rtc else False
        self.policy.recalculate_weight(rtc=rtc)

        actions = self.policy.forward(
            observation_images,
            diffusion_noise,
            inputs["prompt"],
            preprocessed["state_tokens"],
            initial_noise=initial_noise_t,
            copilot_noise=copilot_noise_t,
            velocity_command=velocity_t,
        )

        actions = actions.cpu().float().numpy()
        actions = self.unnormalize_actions(actions, target_dim=32)[:, : self.action_dim]
        return actions


# ------------------------------------------------------------------ #
# ZMQ server                                                           #
# ------------------------------------------------------------------ #

def main():
    port = 5555
    triton_path = "/home/pinhao/realtime-vla/pi05_droid_triton.pkl"
    jax_path = "/home/pinhao/.cache/openpi/openpi-assets/checkpoints/pi05_droid"
    norm_stats_dir = "/home/pinhao/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets/droid"
    tokenizer_path = "google/paligemma-3b-pt-224"
    action_horizon = 15
    action_dim = 8
    noise_step = 5
    default_prompt = "do something"

    print("Initializing model...")
    evaluator = Pi05Evaluator(
        triton_path=triton_path,
        norm_stats_dir=norm_stats_dir,
        tokenizer_path=tokenizer_path,
        action_horizon=action_horizon,
        action_dim=action_dim,
        noise_step=noise_step,
        discrete_state_input=True,
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"[*] Listening on tcp://*:{port}")

    try:
        while True:
            message = socket.recv()
            try:
                data = pickle.loads(message)

                inputs = {
                    "observation/exterior_image_1_left": data["observation/exterior_image_1_left"],
                    "observation/wrist_image_left": data["observation/wrist_image_left"],
                    "observation/joint_position": data["observation/joint_position"],
                    "observation/gripper_position": data["observation/gripper_position"],
                    "prompt": data.get("prompt", default_prompt),
                }
                d = data.get("d", 3)
                s = data.get("s", 8)
                velocity_command = data.get("velocity_command", None)
                proposed_action = data.get("proposed_action", None)

                if proposed_action is None:
                    rtc = False
                    proposed_action = np.random.randn(action_horizon, 32).astype(np.float32)
                else:
                    rtc = True
                    proposed_action = evaluator.normalize_actions(proposed_action, target_dim=32)

                # fill non-robot dims with noise
                proposed_action[:, 8:] = np.random.randn(action_horizon, 24).astype(np.float32)

                # flow-matching forward schedule: build 10 intermediate noise levels
                epsilon = np.random.randn(action_horizon, 32).astype(np.float32)
                initial_noise = np.stack([
                    proposed_action * ((i + 1) / 10) + epsilon * (1 - (i + 1) / 10)
                    for i in range(10)
                ]).astype(np.float32)

                start = time.time()
                actions = evaluator.infer(
                    inputs, proposed_action, initial_noise, velocity_command,
                    rtc=rtc, hajl=False, d=d, s=s,
                )
                # print(f"[+] shape={actions.shape} gripper={actions[:, -1]} time={time.time()-start:.3f}s")

                socket.send(pickle.dumps({"status": "success", "action": actions}))

            except Exception as e:
                err = f"{e}\n{traceback.format_exc()}"
                print(f"[-] {err}")
                socket.send(pickle.dumps({"status": "error", "message": err}))

    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
