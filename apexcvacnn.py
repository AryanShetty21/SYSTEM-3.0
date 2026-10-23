"""Apex-CVA Net (apex_cva_cnn.py)

Custom High-Resolution Convolutional Neural Network + Biomechanical Tracking
Layer
for Clinical Sagittal Posture & CVA Tracking.
"""

import glob
import json
import math
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

LANDMARK_NAMES = [
    "c7_spinous",  # 0: C7 Spinous Process (Cervical Base)
    "tragus",  # 1: Ear Tragus Cartilage
    "eye_canthus",  # 2: Lateral Canthus of Eye
    "acromion",  # 3: Acromion Process of Shoulder
    "hip_trochanter",  # 4: Greater Trochanter of Hip
]
NUM_KEYPOINTS = len(LANDMARK_NAMES)


class CoordinateAttention(nn.Module):

  def __init__(self, in_channels: int, reduction: int = 16):
    super().__init__()
    mid_channels = max(8, in_channels // reduction)
    self.conv1 = nn.Conv2d(
        in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False
    )
    self.bn1 = nn.BatchNorm2d(mid_channels)
    self.act = nn.Hardswish(inplace=True)
    self.conv_h = nn.Conv2d(
        mid_channels, in_channels, kernel_size=1, stride=1, padding=0
    )
    self.conv_w = nn.Conv2d(
        mid_channels, in_channels, kernel_size=1, stride=1, padding=0
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x
    b, c, h, w = x.size()
    x_h = F.adaptive_avg_pool2d(x, (h, 1))
    x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
    y = torch.cat([x_h, x_w], dim=2)
    y = self.act(self.bn1(self.conv1(y)))
    x_h, x_w = torch.split(y, [h, w], dim=2)
    x_w = x_w.permute(0, 1, 3, 2)
    attn_h = torch.sigmoid(self.conv_h(x_h))
    attn_w = torch.sigmoid(self.conv_w(x_w))
    return identity * attn_h * attn_w


class MultiScaleBiomechBlock(nn.Module):

  def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
    super().__init__()
    self.use_residual = (stride == 1) and (in_ch == out_ch)
    mid_ch = out_ch // 2
    self.branch_local = nn.Sequential(
        nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        ),
        nn.BatchNorm2d(in_ch),
        nn.GELU(),
        nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(mid_ch),
    )
    self.branch_context = nn.Sequential(
        nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=in_ch,
            bias=False,
        ),
        nn.BatchNorm2d(in_ch),
        nn.GELU(),
        nn.Conv2d(in_ch, out_ch - mid_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch - mid_ch),
    )
    self.coord_attn = CoordinateAttention(out_ch)
    self.act = nn.GELU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = torch.cat([self.branch_local(x), self.branch_context(x)], dim=1)
    out = self.coord_attn(out)
    if self.use_residual:
      out = out + x
    return self.act(out)


class DSNTSubPixelHead(nn.Module):

  def __init__(
      self,
      in_channels: int,
      num_keypoints: int = NUM_KEYPOINTS,
      temperature: float = 1.0,
  ):
    super().__init__()
    self.num_keypoints = num_keypoints
    self.temperature = temperature
    self.heatmap_conv = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.GELU(),
        nn.Conv2d(in_channels, num_keypoints, kernel_size=1, bias=True),
    )

  def forward(
      self, feat: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_logits = self.heatmap_conv(feat)
    b, k, h, w = raw_logits.shape
    flat_logits = raw_logits.view(b, k, -1) / self.temperature
    prob_map = F.softmax(flat_logits, dim=-1).view(b, k, h, w)

    grid_y = torch.linspace(
        0.0, 1.0, h, device=feat.device, dtype=feat.dtype
    ).view(1, 1, h, 1)
    grid_x = torch.linspace(
        0.0, 1.0, w, device=feat.device, dtype=feat.dtype
    ).view(1, 1, 1, w)

    exp_x = torch.sum(prob_map * grid_x, dim=(2, 3))
    exp_y = torch.sum(prob_map * grid_y, dim=(2, 3))
    coords = torch.stack([exp_x, exp_y], dim=-1)
    confidence = torch.amax(prob_map.view(b, k, -1), dim=-1, keepdim=True)
    return coords, prob_map, confidence


class BiomechanicalTrackingLayer(nn.Module):
  """Learnable osteological constraint layer that refines raw DSNT sub-pixel coordinates."""

  def __init__(self, feat_channels: int = 64, num_keypoints: int = NUM_KEYPOINTS):
    super().__init__()
    in_dim = num_keypoints * 2 + feat_channels
    self.osteological_mlp = nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.GELU(),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Linear(64, num_keypoints * 2),
        nn.Tanh(),
    )
    self.refinement_scale = nn.Parameter(torch.tensor(0.03))

  def forward(
      self, raw_coords: torch.Tensor, fused_feat: torch.Tensor
  ) -> torch.Tensor:
    b, k, _ = raw_coords.shape
    pooled_context = F.adaptive_avg_pool2d(fused_feat, 1).view(b, -1)
    flat_coords = raw_coords.view(b, k * 2)
    joint_input = torch.cat([flat_coords, pooled_context], dim=-1)
    delta = (
        self.osteological_mlp(joint_input).view(b, k, 2) * self.refinement_scale
    )
    return torch.clamp(raw_coords + delta, 0.0, 1.0)


class ApexCVANet(nn.Module):

  def __init__(self, num_keypoints: int = NUM_KEYPOINTS):
    super().__init__()
    self.stem = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.GELU(),
        MultiScaleBiomechBlock(32, 64, stride=2),
    )
    self.stage1 = nn.Sequential(
        MultiScaleBiomechBlock(64, 64, stride=1),
        MultiScaleBiomechBlock(64, 64, stride=1),
    )
    self.stage2 = nn.Sequential(
        MultiScaleBiomechBlock(64, 128, stride=2),
        MultiScaleBiomechBlock(128, 128, stride=1),
        MultiScaleBiomechBlock(128, 128, stride=1),
    )
    self.stage3 = nn.Sequential(
        MultiScaleBiomechBlock(128, 256, stride=2),
        MultiScaleBiomechBlock(256, 256, stride=1),
    )
    self.lat_s3 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
    self.lat_s2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
    self.fusion_block = MultiScaleBiomechBlock(64, 64, stride=1)
    self.head = DSNTSubPixelHead(in_channels=64, num_keypoints=num_keypoints)
    self.biomech_layer = BiomechanicalTrackingLayer(
        feat_channels=64, num_keypoints=num_keypoints
    )

  def forward(
      self, x: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s0 = self.stem(x)
    s1 = self.stage1(s0)
    s2 = self.stage2(s1)
    s3 = self.stage3(s2)
    u3 = F.interpolate(
        self.lat_s3(s3),
        size=s1.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    u2 = F.interpolate(
        self.lat_s2(s2),
        size=s1.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    fused = self.fusion_block(s1 + u2 + u3)
    raw_coords, prob_map, confidence = self.head(fused)
    refined_coords = self.biomech_layer(raw_coords, fused)
    return refined_coords, prob_map, confidence


class BiomechanicalCVALoss(nn.Module):

  def __init__(
      self,
      wing_w: float = 0.05,
      wing_eps: float = 0.01,
      angle_weight: float = 0.30,
  ):
    super().__init__()
    self.w = wing_w
    self.eps = wing_eps
    self.c = self.w - self.w * math.log(1.0 + self.w / self.eps)
    self.angle_weight = angle_weight

  def wing_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target)
    small_err = self.w * torch.log(1.0 + diff / self.eps)
    large_err = diff - self.c
    return torch.where(diff < self.w, small_err, large_err).mean()

  @staticmethod
  def _horizontal_angle_rad(
      base_pt: torch.Tensor, target_pt: torch.Tensor
  ) -> torch.Tensor:
    dy = base_pt[:, 1] - target_pt[:, 1]
    dx = torch.abs(target_pt[:, 0] - base_pt[:, 0]) + 1e-6
    return torch.atan2(dy, dx)

  def forward(
      self, pred_coords: torch.Tensor, target_coords: torch.Tensor
  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_coord = self.wing_loss(pred_coords, target_coords)
    pred_cva = self._horizontal_angle_rad(
        pred_coords[:, 0], pred_coords[:, 1]
    )
    true_cva = self._horizontal_angle_rad(
        target_coords[:, 0], target_coords[:, 1]
    )
    pred_cra = self._horizontal_angle_rad(
        pred_coords[:, 1], pred_coords[:, 2]
    )
    true_cra = self._horizontal_angle_rad(
        target_coords[:, 1], target_coords[:, 2]
    )
    pred_fsa = self._horizontal_angle_rad(
        pred_coords[:, 3], pred_coords[:, 0]
    )
    true_fsa = self._horizontal_angle_rad(
        target_coords[:, 3], target_coords[:, 0]
    )

    loss_angle = (
        F.smooth_l1_loss(pred_cva, true_cva)
        + 0.5 * F.smooth_l1_loss(pred_cra, true_cra)
        + 0.5 * F.smooth_l1_loss(pred_fsa, true_fsa)
    )
    pred_neck_len = torch.norm(pred_coords[:, 1] - pred_coords[:, 0], dim=-1)
    true_neck_len = torch.norm(
        target_coords[:, 1] - target_coords[:, 0], dim=-1
    )
    loss_bone = F.smooth_l1_loss(pred_neck_len, true_neck_len)

    total_loss = loss_coord + self.angle_weight * loss_angle + 0.15 * loss_bone
    cva_mae_deg = torch.mean(torch.abs(pred_cva - true_cva)).item() * (
        180.0 / math.pi
    )
    return total_loss, {
        "total_loss": total_loss.item(),
        "coord_wing_loss": loss_coord.item(),
        "angle_loss": loss_angle.item(),
        "cva_mae_deg": round(cva_mae_deg, 2),
    }


def export_to_onnx(
    model: ApexCVANet, output_path: str = "apex_cva_cnn.onnx"
) -> None:
  model.eval()
  dummy_input = torch.randn(1, 3, 256, 256)
  torch.onnx.export(
      model,
      dummy_input,
      output_path,
      input_names=["input_rgb"],
      output_names=["keypoints_xy", "heatmaps", "confidence"],
      opset_version=14,
      dynamic_axes={
          "input_rgb": {0: "batch_size"},
          "keypoints_xy": {0: "batch_size"},
      },
  )
  print(f"[Apex-CVA Net] Exported ONNX model: {output_path}")
