import torch

from ultralytics import YOLO
import ultralytics.nn.tasks
from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA, Segment, Concat
from ultralytics.utils import loss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import make_divisible
import ultralytics.utils.metrics as metrics

def calculate_nwd(bz1, bz2, constant=12.8):
    """Normalized Gaussian Wasserstein Distance 계산"""
    d2 = (bz1[..., 0] - bz2[..., 0])**2 + (bz1[..., 1] - bz2[..., 1])**2
    wh_distance = (bz1[..., 2] - bz2[..., 2])**2 + (bz1[..., 3] - bz2[..., 3])**2
    w2 = d2 + wh_distance / 4
    return torch.exp(-torch.sqrt(w2 + 1e-7) / constant)

def nwd_iou_loss_patch(box1, box2, **kwargs):
    """수정된 차원 매칭 패치 함수"""
    # 1. 원본 IoU 계산 (결과값 모양: [N, 1])
    iou = loss.original_bbox_iou(box1, box2, **kwargs)

    # 2. NWD 계산을 위한 포맷 변환
    is_xywh = kwargs.get('xywh', False)
    if is_xywh:
        p_xywh, t_xywh = box1, box2
    else:
        p_xywh = box1.clone()
        p_xywh[..., 0:2] = (box1[..., 0:2] + box1[..., 2:4]) / 2
        p_xywh[..., 2:4] = box1[..., 2:4] - box1[..., 0:2]

        t_xywh = box2.clone()
        t_xywh[..., 0:2] = (box2[..., 0:2] + box2[..., 2:4]) / 2
        t_xywh[..., 2:4] = box2[..., 2:4] - box2[..., 0:2]

    # 3. NWD 계산 (결과값 모양: [N])
    nwd = calculate_nwd(p_xywh, t_xywh)

    # 🔥 [핵심 수정] 차원을 [N] -> [N, 1]로 확장하여 iou와 맞춤
    if nwd.ndim < iou.ndim:
        nwd = nwd.unsqueeze(-1)

    # 4. 하이브리드 결합
    alpha = config['nwd_alpha']
    return alpha * nwd + (1 - alpha) * iou

def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()