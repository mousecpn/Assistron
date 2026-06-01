from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

if __package__ in (None, ""):
	sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18

# from .success_dataloader import SuccessFrameDataset


class SuccessClassifier(nn.Module):
	def __init__(self, num_classes: int = 2, pretrained: bool = True):
		super().__init__()
		weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
		self.backbone = resnet18(weights=weights)
		in_features = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(in_features, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.backbone(x)


def build_model(pretrained: bool = True, num_classes: int = 2, device: torch.device | None = None) -> SuccessClassifier:
	model = SuccessClassifier(num_classes=num_classes, pretrained=pretrained)
	if device is not None:
		model = model.to(device)
	return model


def load_checkpoint(model: nn.Module, ckpt_path: str | Path, map_location: str | torch.device = "cpu") -> Dict:
	checkpoint = torch.load(Path(ckpt_path), map_location=map_location)
	state_dict = checkpoint.get("model_state_dict", checkpoint)
	model.load_state_dict(state_dict)
	return checkpoint


def preprocess_bgr_image(image_bgr: np.ndarray, resize_hw: Tuple[int, int] = (224, 224), normalize: bool = True, rgb: bool = True) -> torch.Tensor:
	image = cv2.resize(image_bgr, (int(resize_hw[1]), int(resize_hw[0])), interpolation=cv2.INTER_AREA)
	if rgb:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype(np.float32)
	if normalize:
		image /= 255.0
	image = np.transpose(image, (2, 0, 1))
	return torch.from_numpy(image).float()


@torch.no_grad()
def predict_image(model: nn.Module, image_tensor: torch.Tensor, device: torch.device | None = None) -> Dict[str, float | int]:
	model.eval()
	if image_tensor.ndim == 3:
		image_tensor = image_tensor.unsqueeze(0)
	if device is not None:
		image_tensor = image_tensor.to(device)
	logits = model(image_tensor)
	probs = torch.softmax(logits, dim=1)[0]
	pred = int(torch.argmax(probs).item())
	return {
		"pred": pred,
		"prob_0": float(probs[0].item()),
		"prob_1": float(probs[1].item()),
	}


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def split_indices_by_video(dataset: SuccessFrameDataset, test_size: float, seed: int) -> Tuple[List[int], List[int]]:
	unique_videos = sorted({str(s.video_path) for s in dataset.samples})
	if len(unique_videos) < 2:
		raise ValueError("Need at least 2 videos to create train/test split.")

	rng = random.Random(seed)
	rng.shuffle(unique_videos)

	n_test = max(1, int(round(len(unique_videos) * test_size)))
	n_test = min(n_test, len(unique_videos) - 1)
	test_videos = set(unique_videos[:n_test])

	train_indices: List[int] = []
	test_indices: List[int] = []
	for i, s in enumerate(dataset.samples):
		if str(s.video_path) in test_videos:
			test_indices.append(i)
		else:
			train_indices.append(i)

	if not train_indices or not test_indices:
		raise RuntimeError("Invalid split produced empty train or test indices.")
	return train_indices, test_indices


def make_loader(dataset: SuccessFrameDataset, indices: Sequence[int], batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
	subset = Subset(dataset, list(indices))
	return DataLoader(
		subset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer | None, device: torch.device) -> Dict[str, float]:
	train_mode = optimizer is not None
	model.train(mode=train_mode)

	total_loss = 0.0
	total = 0
	correct = 0

	for images, labels, _ in loader:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		logits = model(images)
		loss = criterion(logits, labels)

		if train_mode:
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

		total_loss += float(loss.item()) * images.size(0)
		preds = torch.argmax(logits, dim=1)
		correct += int((preds == labels).sum().item())
		total += images.size(0)

	avg_loss = total_loss / max(total, 1)
	acc = correct / max(total, 1)
	return {"loss": avg_loss, "acc": acc}


def load_config(config_path: Path) -> Dict:
	with config_path.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f) or {}
	if "train" not in cfg:
		raise ValueError(f"Missing 'train' section in {config_path}")
	return cfg


def main() -> None:
	parser = argparse.ArgumentParser(description="Train success classifier (resnet18 + linear head)")
	parser.add_argument("--config", type=str, default="success_module/config.yaml")
	parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
	args = parser.parse_args()

	config_path = Path(args.config).resolve()
	cfg = load_config(config_path)
	train_cfg = cfg["train"]

	set_seed(int(train_cfg.get("random_seed", 42)))

	dataset = SuccessFrameDataset(config_path)
	train_indices, test_indices = split_indices_by_video(
		dataset=dataset,
		test_size=float(train_cfg.get("test_size", 0.2)),
		seed=int(train_cfg.get("random_seed", 42)),
	)

	batch_size = int(train_cfg.get("batch_size", 32))
	num_workers = int(train_cfg.get("num_workers", 4))
	train_loader = make_loader(dataset, train_indices, batch_size, num_workers, shuffle=True)
	test_loader = make_loader(dataset, test_indices, batch_size, num_workers, shuffle=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SuccessClassifier(num_classes=2, pretrained=(not args.no_pretrained)).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(
		model.parameters(),
		lr=float(train_cfg.get("learning_rate", 1.0e-4)),
		weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
	)

	epochs = int(train_cfg.get("epochs", 10))
	best_test_acc = -1.0

	save_dir = (config_path.parent.parent / str(train_cfg.get("save_dir", "ckpts/success_model"))).resolve()
	save_dir.mkdir(parents=True, exist_ok=True)
	best_ckpt = save_dir / "best.pt"
	last_ckpt = save_dir / "last.pt"
	metrics_path = save_dir / "metrics.json"

	history: List[Dict] = []

	print(f"[INFO] Dataset frames: {len(dataset)}")
	print(f"[INFO] Train frames: {len(train_indices)} | Test frames: {len(test_indices)}")
	print(f"[INFO] Train videos: {len({str(dataset.samples[i].video_path) for i in train_indices})}")
	print(f"[INFO] Test videos: {len({str(dataset.samples[i].video_path) for i in test_indices})}")
	print(f"[INFO] Device: {device}")

	for epoch in range(1, epochs + 1):
		train_stats = run_epoch(model, train_loader, criterion, optimizer, device)
		test_stats = run_epoch(model, test_loader, criterion, optimizer=None, device=device)

		record = {
			"epoch": epoch,
			"train_loss": train_stats["loss"],
			"train_acc": train_stats["acc"],
			"test_loss": test_stats["loss"],
			"test_acc": test_stats["acc"],
		}
		history.append(record)

		print(
			f"[E{epoch:03d}] train_loss={record['train_loss']:.4f} train_acc={record['train_acc']:.4f} "
			f"test_loss={record['test_loss']:.4f} test_acc={record['test_acc']:.4f}"
		)

		if record["test_acc"] > best_test_acc:
			best_test_acc = record["test_acc"]
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"config": cfg,
					"epoch": epoch,
					"test_acc": best_test_acc,
				},
				best_ckpt,
			)

		torch.save(
			{
				"model_state_dict": model.state_dict(),
				"config": cfg,
				"epoch": epoch,
				"test_acc": record["test_acc"],
			},
			last_ckpt,
		)

	with metrics_path.open("w", encoding="utf-8") as f:
		json.dump(
			{
				"best_test_acc": best_test_acc,
				"history": history,
				"train_size": len(train_indices),
				"test_size": len(test_indices),
				"train_videos": sorted({str(dataset.samples[i].video_path) for i in train_indices}),
				"test_videos": sorted({str(dataset.samples[i].video_path) for i in test_indices}),
			},
			f,
			indent=2,
			ensure_ascii=False,
		)

	print(f"[INFO] Best checkpoint: {best_ckpt}")
	print(f"[INFO] Last checkpoint: {last_ckpt}")
	print(f"[INFO] Metrics: {metrics_path}")


if __name__ == "__main__":
	main()
