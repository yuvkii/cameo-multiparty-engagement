"""Head/person detector wrapper around the CrowdHuman-trained YOLOv5m model
bundled with MCGaze (dataExtraction/MCGaze-master/MCGaze_demo/yolo_head),
copied into `yolo_head/` alongside this project so it can run in this
project's own venv (torch/torchvision were added here for this purpose).

Used to locate the robot-person in camera 3's footage: unlike the MediaPipe
FaceLandmarker, this detects heads/bodies regardless of which way they're
facing, so it works even when the robot-person has their back to the camera.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_YOLO_HEAD_DIR = Path(__file__).resolve().parents[2] / "yolo_head"
if str(_YOLO_HEAD_DIR) not in sys.path:
    sys.path.insert(0, str(_YOLO_HEAD_DIR))

from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_coords  # noqa: E402


def _load_model(weights_path: str, map_location: torch.device):
    # The bundled 2021-era checkpoint pickles the full model object (not just a
    # state_dict); torch >=2.6 defaults torch.load to weights_only=True, which
    # breaks that. This is a known, widely-used public checkpoint (CrowdHuman
    # YOLOv5m head detector) so unpickling it is trusted here. attempt_load()
    # calls torch.load internally, so the patch must be active for that call.
    orig_torch_load = torch.load
    try:
        torch.load = lambda *a, **kw: orig_torch_load(*a, **{**kw, "weights_only": False})
        return attempt_load(weights_path, map_location=map_location)
    finally:
        torch.load = orig_torch_load

HEAD_CLASS_ID = 1
PERSON_CLASS_ID = 0


@dataclass(slots=True)
class Detection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class HeadDetector:
    """CrowdHuman YOLOv5m head/person detector for a single BGR frame at a time."""

    def __init__(
        self,
        weights_path: str | Path,
        img_size: int = 640,
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = _load_model(str(weights_path), map_location=self.device)
        self.model.eval()
        self.stride = int(self.model.stride.max())
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = self.model.names

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        img = letterbox(frame_bgr, new_shape=self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        pred = self.model(tensor, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

        detections: list[Detection] = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(tensor.shape[2:], pred[:, :4], frame_bgr.shape).round()
            for *xyxy, conf, cls in pred.tolist():
                class_id = int(cls)
                detections.append(
                    Detection(
                        bbox_xyxy=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        confidence=float(conf),
                        class_id=class_id,
                        class_name=self.names[class_id],
                    )
                )
        return detections

    def detect_heads(self, frame_bgr: np.ndarray) -> list[Detection]:
        return [d for d in self.detect(frame_bgr) if d.class_id == HEAD_CLASS_ID]
