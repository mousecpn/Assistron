from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
from typing import Dict, List, Optional, Union

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from .success_model import build_model, load_checkpoint, predict_image, preprocess_bgr_image

_DEFAULT_CKPT = Path(__file__).resolve().parent / "best_both.pt"


def load_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return preprocess_bgr_image(image)


class SuccessDetector:
    """Wrapper around SuccessClassifier for convenient single/batch inference.

    Args:
        ckpt_path: Path to the checkpoint file. Defaults to
            ``interaction_detection/best.pt``.
        device: torch device string or ``torch.device``. Defaults to CUDA if
            available, otherwise CPU.
    """

    def __init__(
        self,
        ckpt_path: Union[str, Path] = _DEFAULT_CKPT,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.model = build_model(pretrained=True, device=self.device)
        load_checkpoint(self.model, Path(ckpt_path).resolve(), map_location=self.device)
        self.model.eval()

        # Async inference state
        self.results: Optional[List[Dict[str, Union[int, float]]]] = None
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_batch(
        self,
        images: Union[torch.Tensor, List[np.ndarray]],
    ) -> List[Dict[str, Union[int, float]]]:
        """Run inference on a batch of images.

        Args:
            images: Either a pre-processed ``torch.Tensor`` of shape
                ``(N, C, H, W)`` (float32, values in [0, 1]), **or** a list of
                BGR ``np.ndarray`` images (HxWxC, uint8) that will be
                pre-processed automatically.

        Returns:
            A list of dicts, one per image, each containing:
            ``{"pred": int, "prob_0": float, "prob_1": float}``.
        """
        if isinstance(images, list):
            tensors = [preprocess_bgr_image(img, rgb=True) for img in images]
            batch = torch.stack(tensors, dim=0)
        else:
            batch = images
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)

        batch = batch.to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)  # (N, 2)
        preds = probs[:, 1] > 0.3

        results: List[Dict[str, Union[int, float]]] = []
        for i in range(probs.shape[0]):
            results.append({
                "pred": int(preds[i].item()),
                "prob_0": float(probs[i, 0].item()),
                "prob_1": float(probs[i, 1].item()),
            })
        return results

    def predict_batch_async(self, images: Union[torch.Tensor, List[np.ndarray]]) -> None:
        """Non-blocking version of :meth:`predict_batch`.

        Launches inference in a background thread. Returns immediately.
        When inference finishes, ``self.results`` is populated and
        ``self.is_ready()`` returns ``True``.

        Call ``self.wait()`` to block until results are available.
        """
        self._ready.clear()
        # self.results = None

        def _run() -> None:
            self.results = self.predict_batch(images)
            self._ready.set()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def is_ready(self) -> bool:
        """Return ``True`` if async results are available."""
        return self._ready.is_set()

    def wait(self, timeout: Optional[float] = None) -> Optional[List[Dict[str, Union[int, float]]]]:
        """Block until async inference completes (or *timeout* seconds elapse).

        Returns ``self.results`` (``None`` if timed out).
        """
        self._ready.wait(timeout=timeout)
        return self.results

    def predict_bgr(self, image_bgr: np.ndarray) -> Dict[str, Union[int, float]]:
        """Convenience method for a single BGR image (HxWxC np.ndarray)."""
        return self.predict_batch([image_bgr])[0]

    def predict_path(self, image_path: Union[str, Path]) -> Dict[str, Union[int, float]]:
        """Convenience method that reads an image from *image_path* and runs inference."""
        return self.predict_batch(load_image(Path(image_path)).unsqueeze(0))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer success model on a single image")
    parser.add_argument("--ckpt", type=str, default=str(_DEFAULT_CKPT))
    parser.add_argument("--image", type=str, required=True, help="Image path to run inference on")
    args = parser.parse_args()

    detector = SuccessDetector(ckpt_path=args.ckpt)
    print(f"[INFO] Loaded checkpoint: {args.ckpt}")
    print(f"[INFO] Device: {detector.device}")

    image_path = Path(args.image).resolve()
    result = detector.predict_path(image_path)
    print(f"[RESULT] image={image_path}")
    print(result)


if __name__ == "__main__":
    main()
