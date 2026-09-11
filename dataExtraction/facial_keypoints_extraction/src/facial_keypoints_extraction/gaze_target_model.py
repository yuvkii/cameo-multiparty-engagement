"""Wrapper around Gaze-LLE (github.com/fkryan/gazelle), a calibration-free
gaze-target model: given an image and a person's head box, it predicts a
heatmap over that same image of where they're looking. No camera intrinsics,
no distortion model, no depth — just RGB pixels in, screen position out.

This is what the wide-angle mocap cameras (Cam_2, Cam_3, ... under a 6cams
rig) use for gaze-target detection, since their calibration data doesn't
match OpenCV's distortion model closely enough for angle-based geometry (see
memory: camera3_robot_detection_pipeline / gaze_target_pipeline for the full
story of why the calibrated-angle approach was abandoned in favour of this).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(slots=True)
class GazeTargetResult:
    heatmap: np.ndarray  # (out_h, out_w), values in [0, 1]
    peak_xy: tuple[int, int]  # pixel coords (in the ORIGINAL frame) of the heatmap peak
    inout_score: float | None  # probability gaze target is within frame, if available


class GazeTargetModel:
    """Loads Gaze-LLE once; call detect_targets() per frame with a list of
    head boxes (in original-frame pixel xyxy) to get a gaze result per head."""

    def __init__(self, model_name: str = "gazelle_dinov2_vitb14", device: str = "cpu") -> None:
        # This model (DINOv2 ViT-B backbone) dominates the pipeline's runtime
        # and used to run on CPU unconditionally, with no way to ask for
        # anything else. Measured on 2048x1088 frames: ~2675ms/frame on CPU
        # vs ~24ms on an RTX 5080 -- the difference between a day and an hour
        # for a full 6-session date. Kept defaulting to "cpu" so existing
        # callers are unaffected.
        self.device = torch.device(device)
        self.model, self.transform = torch.hub.load("fkryan/gazelle", model_name, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect_targets(
        self, frame_rgb: np.ndarray, head_boxes_xyxy: list[tuple[int, int, int, int]]
    ) -> list[GazeTargetResult]:
        if not head_boxes_xyxy:
            return []

        h, w = frame_rgb.shape[:2]
        img_tensor = self.transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
        norm_bboxes = [(x0 / w, y0 / h, x1 / w, y1 / h) for x0, y0, x1, y1 in head_boxes_xyxy]

        output = self.model({"images": img_tensor, "bboxes": [norm_bboxes]})
        heatmaps = output["heatmap"][0]
        inout = output["inout"][0] if output["inout"] is not None else None

        results = []
        for i in range(len(head_boxes_xyxy)):
            heatmap = heatmaps[i].cpu().numpy()
            peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            peak_px = (int(peak_idx[1] / heatmap.shape[1] * w), int(peak_idx[0] / heatmap.shape[0] * h))
            inout_score = float(inout[i]) if inout is not None else None
            results.append(GazeTargetResult(heatmap=heatmap, peak_xy=peak_px, inout_score=inout_score))
        return results
