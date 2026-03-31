import pandas as pd
from pathlib import Path
from PIL import Image
import random
import shutil
from typing import List, Dict
from tqdm import tqdm
from tqdm.auto import tqdm

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ultralytics import YOLO
import ultralytics.nn.tasks
from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA, Segment, Concat
from ultralytics.utils import loss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import make_divisible
import ultralytics.utils.metrics as metrics

from loss import *
from caf import *


def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

if not hasattr(loss, 'original_bbox_iou'):
    loss.original_bbox_iou = loss.bbox_iou

# 패치 적용
loss.bbox_iou = nwd_iou_loss_patch
print("✅ NWD Loss 패치가 수정되어 성공적으로 주입되었습니다.")

print("\n" + "="*60)
print(f"🔥 Expert Mode Training Start: {MODEL_NAME}")
print(f"📌 Settings: Optimizer={args['optimizer']}, LR={args['lr0']}")
print("="*60)

# **args로 딕셔너리 풀어서 전달
results = model.train(**args)