from collections import defaultdict
import cv2
from datetime import datetime
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
import pandas as pd
from pathlib import Path
from PIL import Image
import random
import shutil
from typing import List, Dict
from tqdm import tqdm
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
import zipfile


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

print("\n" + "="*60)
print(f"🔥 Expert Mode Training Start: {MODEL_NAME}")
print(f"📌 Settings: Optimizer={args['optimizer']}, LR={args['lr0']}")
print("="*60)

# **args로 딕셔너리 풀어서 전달
results = model.train(**args)