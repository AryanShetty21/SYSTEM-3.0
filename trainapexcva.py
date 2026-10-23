"""train_apex_cva.py — Automated Hugging Face Teacher-Student Distillation & Biomechanical Training Pipeline."""

import argparse
import glob
import json
import math
import os
import random
from typing import Tuple

from apex_cva_cnn import (
    ApexCVANet,
    BiomechanicalCVALoss,
    LANDMARK_NAMES,
    export_to_onnx,
)
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import DataLoader, Dataset


def auto_label_with_huggingface_teacher(
    raw_photos_dir: str,
    output_labels_dir: str,
    hf_model_id: str = "usyd-community/vitpose-base-simple",
) -> int:
  image_paths = []
  for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
    image_paths.extend(glob.glob(os.path.join(raw_photos_dir, ext)))
  if not image_paths:
    return 0

  try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation
  except ImportError:
    print("[!] 'transformers' not installed. Run: pip install transformers")
    return 0

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"[HF Teacher] Loading '{hf_model_id}' on {device}...")
  processor = AutoProcessor.from_pretrained(hf_model_id)
  teacher = (
      VitPoseForPoseEstimation.from_pretrained(hf_model_id).to(device).eval()
  )

  os.makedirs(output_labels_dir, exist_ok=True)
  generated = 0

  for img_path in sorted(image_paths):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_json = os.path.join(output_labels_dir, f"hf_auto_{base_name}.json")
    if os.path.exists(out_json):
      continue

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    boxes = [[[0.0, 0.0, float(w), float(h)]]]
    inputs = processor(images=img, boxes=boxes, return_tensors="pt").to(device)

    with torch.no_grad():
      outputs = teacher(**inputs)
    pose_res = processor.post_process_pose_estimation(outputs, boxes=boxes)[0][
        0
    ]
    kps = pose_res["keypoints"].cpu().numpy()
    scores = pose_res["scores"].cpu().numpy()

    use_right = (scores[4] + scores[6]) >= (scores[3] + scores[5])
    ear = kps[4] if use_right else kps[3]
    eye = kps[2] if use_right else kps[1]
    acromion = kps[6] if use_right else kps[5]
    hip = kps[12] if use_right else kps[11]

    ear_x, ear_y = float(ear[0] / w), float(ear[1] / h)
    eye_x, eye_y = float(eye[0] / w), float(eye[1] / h)
    sh_x, sh_y = float(acromion[0] / w), float(acromion[1] / h)
    hip_x, hip_y = float(hip[0] / w), float(hip[1] / h)

    facing_right = eye_x >= ear_x
    posterior_dir = -1.0 if facing_right else 1.0
    neck_vert = abs(sh_y - ear_y)

    tragus_x = max(0.0, min(1.0, ear_x + (eye_x - ear_x) * 0.15))
    tragus_y = max(0.0, min(1.0, ear_y + (eye_y - ear_y) * 0.15))
    c7_x = max(0.0, min(1.0, sh_x + posterior_dir * (neck_vert * 0.18)))
    c7_y = max(0.0, min(1.0, sh_y - (neck_vert * 0.30)))

    annotation = {
        "image_path": os.path.abspath(img_path),
        "image_width": w,
        "image_height": h,
        "source": f"hf_teacher:{hf_model_id}",
        "keypoints": {
            "c7_spinous": {"x": round(c7_x, 6), "y": round(c7_y, 6)},
            "tragus": {"x": round(tragus_x, 6), "y": round(tragus_y, 6)},
            "eye_canthus": {"x": round(eye_x, 6), "y": round(eye_y, 6)},
            "acromion": {"x": round(sh_x, 6), "y": round(sh_y, 6)},
            "hip_trochanter": {"x": round(hip_x, 6), "y": round(hip_y, 6)},
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
      json.dump(annotation, f, indent=2)
    generated += 1

  print(f"[HF Teacher] Auto-generated {generated} labels.")
  return generated


class AugmentedPostureDataset(Dataset):

  def __init__(self, data_dir: str, image_size: int = 256, augment: bool = True):
    self.json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    self.data_dir = data_dir
    self.image_size = image_size
    self.augment = augment

  def __len__(self) -> int:
    return len(self.json_files)

  def _rotate_point(
      self, x: float, y: float, angle_deg: float
  ) -> Tuple[float, float]:
    rad = math.radians(angle_deg)
    cx, cy = 0.5, 0.5
    dx, dy = x - cx, y - cy
    nx = cx + dx * math.cos(rad) - dy * math.sin(rad)
    ny = cy + dx * math.sin(rad) + dy * math.cos(rad)
    return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(self.json_files[idx], "r", encoding="utf-8") as f:
      rec = json.load(f)

    img_path = rec["image_path"]
    if not os.path.isabs(img_path):
      img_path = os.path.join(self.data_dir, os.path.basename(img_path))

    img = (
        Image.open(img_path)
        .convert("RGB")
        .resize((self.image_size, self.image_size))
    )
    coords = [
        [float(rec["keypoints"][k]["x"]), float(rec["keypoints"][k]["y"])]
        for k in LANDMARK_NAMES
    ]

    if self.augment:
      if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        coords = [[1.0 - pt[0], pt[1]] for pt in coords]
      if random.random() < 0.7:
        angle = random.uniform(-8.0, 8.0)
        img = img.rotate(-angle, resample=Image.BILINEAR)
        coords = [list(self._rotate_point(pt[0], pt[1], angle)) for pt in coords]
      if random.random() < 0.6:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

    img_np = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1), torch.tensor(
        coords, dtype=torch.float32
    )


def train_model(
    data_dir: str,
    raw_photos_dir: str = "./raw_photos",
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 1e-3,
    onnx_out: str = "apex_cva_cnn.onnx",
) -> None:
  if os.path.isdir(raw_photos_dir):
    auto_label_with_huggingface_teacher(raw_photos_dir, data_dir)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = ApexCVANet().to(device)
  criterion = BiomechanicalCVALoss(wing_w=0.05, wing_eps=0.01, angle_weight=0.30)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs
  )

  dataset = AugmentedPostureDataset(data_dir=data_dir, augment=True)
  if len(dataset) == 0:
    print(f"[!] No labeled samples yet in '{data_dir}'. Exporting baseline {onnx_out}...")
    export_to_onnx(model.cpu(), onnx_out)
    return

  loader = DataLoader(
      dataset, batch_size=min(batch_size, len(dataset)), shuffle=True
  )
  best_mae = float("inf")

  for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss, epoch_cva_mae = 0.0, 0.0
    for imgs, targets in loader:
      imgs, targets = imgs.to(device), targets.to(device)
      optimizer.zero_grad()
      pred_coords, _, _ = model(imgs)
      loss, stats = criterion(pred_coords, targets)
      loss.backward()
      optimizer.step()
      epoch_loss += stats["total_loss"]
      epoch_cva_mae += stats["cva_mae_deg"]

    scheduler.step()
    avg_mae = epoch_cva_mae / len(loader)
    if avg_mae < best_mae:
      best_mae = avg_mae
      torch.save(model.state_dict(), "apex_cva_best.pth")

    if epoch % 10 == 0 or epoch == 1:
      print(
          f"Epoch [{epoch:03d}/{epochs:03d}] | Mean CVA Error: {avg_mae:.2f}° (Best: {best_mae:.2f}°)"
      )

  model.load_state_dict(torch.load("apex_cva_best.pth", map_location="cpu"))
  export_to_onnx(model.cpu(), onnx_out)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw_photos_dir", type=str, default="./raw_photos")
  parser.add_argument("--data_dir", type=str, default="./labels")
  parser.add_argument("--epochs", type=int, default=80)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--onnx_out", type=str, default="apex_cva_cnn.onnx")
  args = parser.parse_args()
  os.makedirs(args.data_dir, exist_ok=True)
  os.makedirs(args.raw_photos_dir, exist_ok=True)
  train_model(
      args.data_dir,
      args.raw_photos_dir,
      args.epochs,
      args.batch_size,
      args.lr,
      args.onnx_out,
  )
