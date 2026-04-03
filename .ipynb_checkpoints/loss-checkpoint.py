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

def nwd_iou_loss_patch(box1, box2, **kwargs):
    """
    YOLO의 기본 IoU 연산에 NWD를 결합하는 하이브리드 손실 함수 패치.
    """
    
    # 1. 원본 IoU 계산 (CIoU, DIoU 등 YOLO 설정값)
    # YOLO의 내부 엔진을 그대로 사용하여 기존의 검증된 성능을 유지.
    # 결과값(iou) 모양: 보통 [N, 1] (N은 매칭된 Bbox 쌍의 개수)
    iou = loss.original_bbox_iou(box1, box2, **kwargs)

    # 2. NWD 계산을 위한 좌표 포맷 변환 (XYXY -> XYWH)
    # calculate_nwd 함수는 중심점(cx, cy)과 크기(w, h)를 기준으로 동작하기 때문.
    is_xywh = kwargs.get('xywh', False)
    
    if is_xywh:
        # 이미 중심점 기반 포맷이라면 그대로 사용
        p_xywh, t_xywh = box1, box2
    else:
        # [x1, y1, x2, y2] (좌상단, 우상단) -> [cx, cy, w, h] (중심, 크기) 변환
        # 원본 데이터 훼손 방지를 위해 .clone() 사용
        p_xywh = box1.clone()
        # [..., 0:2]는 x1, y1 | [..., 2:4]는 x2, y2
        # 중심점 (cx, cy) = (좌상단 + 우하단) / 2
        p_xywh[..., 0:2] = (box1[..., 0:2] + box1[..., 2:4]) / 2
        # 크기 (w, h) = (우하단 - 좌상단)
        p_xywh[..., 2:4] = box1[..., 2:4] - box1[..., 0:2]

        t_xywh = box2.clone()
        t_xywh[..., 0:2] = (box2[..., 0:2] + box2[..., 2:4]) / 2
        t_xywh[..., 2:4] = box2[..., 2:4] - box2[..., 0:2]

    # 3. NWD(가우시안 거리 유사도) 계산
    # 결과값(nwd) 모양: [N] (1차원 벡터)
    nwd = calculate_nwd(p_xywh, t_xywh)

    # 4. 차원 정렬 (Broadcasting 에러 방지)
    # iou는 [N, 1]이고 nwd는 [N]일 경우, 덧셈 연산 시 차원 불일치 에러가 발생할 수 있음.
    # unsqueeze(-1)을 통해 nwd를 [N, 1]로 확장하여 iou와 모양을 맞춤.
    if nwd.ndim < iou.ndim:
        nwd = nwd.unsqueeze(-1)

    # 5. 하이브리드 결합 (Alpha Blending)
    # alpha 값에 따라 두 손실 함수의 비중을 조절.
    # alpha=1.0 이면 순수 NWD만 사용, alpha=0.0 이면 기존 IoU만 사용.
    alpha = config['nwd_alpha']
    return alpha * nwd + (1 - alpha) * iou

def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()