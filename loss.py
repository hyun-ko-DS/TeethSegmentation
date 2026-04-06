import json
from pathlib import Path

import torch
import ultralytics.nn.tasks
import ultralytics.utils.metrics as metrics
from ultralytics import YOLO
from ultralytics.nn.modules import C2PSA, C3k2, Concat, Conv, SPPF, Segment
from ultralytics.utils import loss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import make_divisible


def calculate_nwd(p_box, t_box, constant=12.8):
    """
    Calculates the Normalized Gaussian Wasserstein Distance (NWD).
    
    NWD models bounding boxes as 2D Gaussian distributions and calculates 
    the distance between them. This is particularly effective for small 
    object detection (like teeth) where IoU might be zero.

    Args:
        p_box (Tensor): Predicted boxes in [x_center, y_center, width, height] format (N, 4).
        t_box (Tensor): Target boxes in [x_center, y_center, width, height] format (N, 4).
        constant (float): Normalization constant. Lower values increase sensitivity to distance.
    
    Returns:
        Tensor: NWD values between 0 and 1 (1 indicates a perfect match).
    """
    # 1. Calculate squared Euclidean distance between centroids (d^2)
    # indices: [0]=cx, [1]=cy
    d2 = (p_box[..., 0] - t_box[..., 0])**2 + (p_box[..., 1] - t_box[..., 1])**2
    
    # 2. Calculate the difference in Gaussian variance (width and height)
    # Based on the Wasserstein distance formula for 2D Gaussians
    wh_distance = (p_box[..., 2] - t_box[..., 2])**2 + (p_box[..., 3] - t_box[..., 3])**2
    
    # 3. Final form of the squared 2nd-order Wasserstein distance (W2^2)
    w2 = d2 + wh_distance / 4
    
    # 4. Normalize to a 0~1 range using an exponential function
    # sqrt(w2) provides the actual distance; divided by 'constant' to adjust sensitivity
    return torch.exp(-torch.sqrt(w2 + 1e-7) / constant)


def make_nwd_iou_loss_patch(nwd_alpha: float):
    """
    Creates a patched version of bbox_iou that blends standard IoU with NWD.

    Args:
        nwd_alpha (float): The blending ratio (0.0 to 1.0). 
                           Higher alpha gives more weight to NWD.
    
    Returns:
        function: A patched_bbox_iou function compatible with Ultralytics loss modules.
    """

    def patched_bbox_iou(box1, box2, **kwargs):
        # Calculate standard IoU using the original Ultralytics function
        iou = loss.original_bbox_iou(box1, box2, **kwargs)

        is_xywh = kwargs.get("xywh", False)

        # Ensure boxes are in [cx, cy, w, h] format for NWD calculation
        if is_xywh:
            p_xywh, t_xywh = box1, box2
        else:
            # Convert [x1, y1, x2, y2] to [cx, cy, w, h]
            p_xywh = box1.clone()
            p_xywh[..., 0:2] = (box1[..., 0:2] + box1[..., 2:4]) / 2
            p_xywh[..., 2:4] = box1[..., 2:4] - box1[..., 0:2]

            t_xywh = box2.clone()
            t_xywh[..., 0:2] = (box2[..., 0:2] + box2[..., 2:4]) / 2
            t_xywh[..., 2:4] = box2[..., 2:4] - box2[..., 0:2]

        # Calculate NWD
        nwd = calculate_nwd(p_xywh, t_xywh)

        # Match dimensions with IoU tensor if necessary
        if nwd.ndim < iou.ndim:
            nwd = nwd.unsqueeze(-1)

        # Blend IoU and NWD using the provided alpha
        # Formula: alpha * NWD + (1 - alpha) * IoU
        loss_value = nwd_alpha * nwd + (1 - nwd_alpha) * iou
        
        # Note: Printing here may slow down training. Use only for debugging.
        # print(f"DEBUG: Blended Loss Value: {loss_value.mean().item():.4f}")
        
        return loss_value

    return patched_bbox_iou