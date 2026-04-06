import json
from pathlib import Path

import torch
from ultralytics import YOLO
import ultralytics.nn.tasks
from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA, Segment, Concat
from ultralytics.utils import loss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import make_divisible
import ultralytics.utils.metrics as metrics

def calculate_nwd(p_box, t_box, constant=12.8):
    """
    Normalized Gaussian Wasserstein Distance 계산

    Args:
        p_box (Tensor): 예측된 [x_center, y_center, width, height] (N, 4)
        t_box (Tensor): 타겟 [x_center, y_center, width, height] (N, 4)
        constant (float): 정규화 계수. 작을수록 거리 변화에 민감하게 반응 (치아 탐지에 최적화)
    
    Returns:
        Tensor: 0~1 사이의 NWD 값 (1에 가까울수록 두 박스가 일치)

    인덱스 상세 설명
    - [..., 0]: cx (박스 중심의 가로 좌표)
    - [..., 1]: cy (박스 중심의 세로 좌표)
    - [..., 2]:  w (박스 전체 너비)
    - [..., 3]:  h (박스 전체 높이)
    """
    # 1. 중심점 (cx, cy) 간의 유클리드 거리 제곱 계산 (d^2)
    d2 = (p_box[..., 0] - t_box[..., 0])**2 + (p_box[..., 1] - t_box[..., 1])**2
    
    # 2. 가우시안 분포의 분산(너비, 높이) 차이 계산
    # Wasserstein 거리 공식에 따라 wh 차이의 합을 4로 나눔
    wh_distance = (p_box[..., 2] - t_box[..., 2])**2 + (p_box[..., 3] - t_box[..., 3])**2
    
    # 3. 2차 Wasserstein 거리 제곱(W2^2)의 최종 형태
    w2 = d2 + wh_distance / 4
    
    # 4. 지수 함수를 이용한 0~1 사이 값으로 정규화
    # sqrt(w2)를 통해 거리를 구하고, 상수(constant)로 나누어 민감도를 조절함
    return torch.exp(-torch.sqrt(w2 + 1e-7) / constant)

def make_nwd_iou_loss_patch(nwd_alpha: float):
    """
    Build a bbox_iou replacement that blends IoU with NWD using the given ``nwd_alpha``.

    ``nwd_alpha`` must be supplied by the caller (e.g. train.py from the per-model config);
    this module does not load config.json for it.
    """

    def patched_bbox_iou(box1, box2, **kwargs):
        iou = loss.original_bbox_iou(box1, box2, **kwargs)

        is_xywh = kwargs.get("xywh", False)

        if is_xywh:
            p_xywh, t_xywh = box1, box2
        else:
            p_xywh = box1.clone()
            p_xywh[..., 0:2] = (box1[..., 0:2] + box1[..., 2:4]) / 2
            p_xywh[..., 2:4] = box1[..., 2:4] - box1[..., 0:2]

            t_xywh = box2.clone()
            t_xywh[..., 0:2] = (box2[..., 0:2] + box2[..., 2:4]) / 2
            t_xywh[..., 2:4] = box2[..., 2:4] - box2[..., 0:2]

        nwd = calculate_nwd(p_xywh, t_xywh)

        if nwd.ndim < iou.ndim:
            nwd = nwd.unsqueeze(-1)

        alpha = nwd_alpha
        loss_function = alpha * nwd + (1 - alpha) * iou
        print(f"Loss function: {loss_function}")
        return loss_function

    return patched_bbox_iou