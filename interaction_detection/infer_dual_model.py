from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import threading
from typing import Dict, List, Optional, Union

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from .success_model import build_model, load_checkpoint, preprocess_bgr_image

_DEFAULT_CKPT_OPEN = Path(__file__).resolve().parent / "best_open_0.83.pt"
_DEFAULT_CKPT_CLOSE = Path(__file__).resolve().parent / "best_close_0.85.pt"


def load_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return preprocess_bgr_image(image)


class DualSuccessDetector:
    """Success detector that selects the model based on gripper state.

    - When gripper is **closed**, uses the ``best_open`` model to detect
      whether the gripper is about to open (i.e. task success).
    - When gripper is **open**, uses the ``best_close`` model to detect
      whether the gripper is about to close (i.e. task success).

    Args:
        ckpt_open: Path to checkpoint used when gripper is closed.
            Defaults to ``best_open_10_0.73.pt``.
        ckpt_close: Path to checkpoint used when gripper is open.
            Defaults to ``best_close_01_0.82.pt``.
        device: torch device string or ``torch.device``. Defaults to CUDA if
            available, otherwise CPU.
    """

    def __init__(
        self,
        ckpt_open: Union[str, Path] = _DEFAULT_CKPT_OPEN,
        ckpt_close: Union[str, Path] = _DEFAULT_CKPT_CLOSE,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.model_open = build_model(pretrained=True, device=self.device)
        load_checkpoint(self.model_open, Path(ckpt_open).resolve(), map_location=self.device)
        self.model_open.eval()

        self.model_close = build_model(pretrained=True, device=self.device)
        load_checkpoint(self.model_close, Path(ckpt_close).resolve(), map_location=self.device)
        self.model_close.eval()

        # Async inference state — single persistent worker thread
        self.results: Optional[List[Dict[str, Union[int, float]]]] = None
        self._ready = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        """Persistent background thread that processes inference requests."""
        while True:
            item = self._queue.get()
            if item is None:  # sentinel: shut down
                break
            images, gripper_closed = item
            self.results = self.predict_batch(images, gripper_closed)
            self._ready.set()

    def _select_model(self, gripper_closed: bool) -> torch.nn.Module:
        """Return the appropriate model given the current gripper state."""
        # gripper closed -> use open model (detects opening transition)
        # gripper open   -> use close model (detects closing transition)
        return self.model_open if gripper_closed else self.model_close

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_batch(
        self,
        images: Union[torch.Tensor, List[np.ndarray]],
        gripper_closed: bool,
    ) -> List[Dict[str, Union[int, float]]]:
        """Run inference on a batch of images using the model appropriate for
        the current gripper state.

        Args:
            images: Either a pre-processed ``torch.Tensor`` of shape
                ``(N, C, H, W)`` (float32, values in [0, 1]), **or** a list of
                BGR ``np.ndarray`` images (HxWxC, uint8) that will be
                pre-processed automatically.
            gripper_closed: ``True`` if the gripper is currently closed;
                ``False`` if it is open.

        Returns:
            A list of dicts, one per image, each containing:
            ``{"pred": int, "prob_0": float, "prob_1": float}``.
        """
        model = self._select_model(gripper_closed)

        if isinstance(images, list):
            tensors = [preprocess_bgr_image(img, rgb=True) for img in images]
            batch = torch.stack(tensors, dim=0)
        else:
            batch = images
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)

        batch = batch.to(self.device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)  # (N, 2)
        preds = probs[:, 1] > 0.5

        results: List[Dict[str, Union[int, float]]] = []
        for i in range(probs.shape[0]):
            results.append({
                "pred": int(preds[i].item()),
                "prob_0": float(probs[i, 0].item()),
                "prob_1": float(probs[i, 1].item()),
            })
        return results

    def predict_batch_async(
        self,
        images: Union[torch.Tensor, List[np.ndarray]],
        gripper_closed: bool,
    ) -> None:
        """Non-blocking version of :meth:`predict_batch`.

        Launches inference in a background thread. Returns immediately.
        When inference finishes, ``self.results`` is populated and
        ``self.is_ready()`` returns ``True``.

        Call ``self.wait()`` to block until results are available.
        """
        self._ready.clear()
        # Discard any pending job that hasn't started yet, then submit new one.
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put((images, gripper_closed))

    def is_ready(self) -> bool:
        """Return ``True`` if async results are available."""
        return self._ready.is_set()

    def wait(self, timeout: Optional[float] = None) -> Optional[List[Dict[str, Union[int, float]]]]:
        """Block until async inference completes (or *timeout* seconds elapse).

        Returns ``self.results`` (``None`` if timed out).
        """
        self._ready.wait(timeout=timeout)
        return self.results

    def predict_bgr(
        self,
        image_bgr: np.ndarray,
        gripper_closed: bool,
    ) -> Dict[str, Union[int, float]]:
        """Convenience method for a single BGR image (HxWxC np.ndarray)."""
        return self.predict_batch([image_bgr], gripper_closed)[0]

    def predict_path(
        self,
        image_path: Union[str, Path],
        gripper_closed: bool,
    ) -> Dict[str, Union[int, float]]:
        """Convenience method that reads an image from *image_path* and runs inference."""
        return self.predict_batch(load_image(Path(image_path)).unsqueeze(0), gripper_closed)[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer dual success model (selects model based on gripper state)"
    )
    parser.add_argument("--ckpt_open", type=str, default=str(_DEFAULT_CKPT_OPEN),
                        help="Checkpoint used when gripper is closed (detects open transition)")
    parser.add_argument("--ckpt_close", type=str, default=str(_DEFAULT_CKPT_CLOSE),
                        help="Checkpoint used when gripper is open (detects close transition)")
    parser.add_argument("--image", type=str, required=True, help="Image path to run inference on")
    parser.add_argument("--gripper_closed", action="store_true",
                        help="Set this flag if the gripper is currently closed")
    args = parser.parse_args()

    detector = DualSuccessDetector(ckpt_open=args.ckpt_open, ckpt_close=args.ckpt_close)
    print(f"[INFO] Device: {detector.device}")
    print(f"[INFO] Gripper state: {'closed' if args.gripper_closed else 'open'}")
    print(f"[INFO] Using model: {'best_open (gripper closed)' if args.gripper_closed else 'best_close (gripper open)'}")

    image_path = Path(args.image).resolve()
    result = detector.predict_path(image_path, gripper_closed=args.gripper_closed)
    print(f"[RESULT] image={image_path}")
    print(result)


if __name__ == "__main__":
    main()
