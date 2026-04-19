# utils/selective_search.py
import cv2
import torch
import numpy as np


def get_selective_search_rois(image_tensor, max_rois=2000):
    """
    使用 OpenCV 的 Selective Search 提取候选框
    :param image_tensor: 输入图像 [C, H, W] 的 Tensor 或 numpy array
    :param max_rois: 最大保留的候选框数量
    :return: 候选框 Tensor, 形状为 [N, 4], 坐标格式 (x1, y1, x2, y2)
    """
    # 将 PyTorch Tensor [C, H, W] 转换为 OpenCV 格式 [H, W, C] (uint8)
    if isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        # RGB 转 BGR (OpenCV 默认)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_cv = image_tensor

    # 初始化 Selective Search
    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_cv)
    # 采用快速模式 (也可以选择 'quality' 高质量模式)
    ss.switchToSelectiveSearchFast()

    # 提取框 (输出格式为 x, y, w, h)
    rects = ss.process()

    # 转换为 (x1, y1, x2, y2) 格式并截断最大数量
    rois = []
    for (x, y, w, h) in rects[:max_rois]:
        rois.append([x, y, x + w, y + h])

    return torch.tensor(rois, dtype=torch.float32)