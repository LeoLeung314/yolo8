from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray
import torch
import torchvision

# image suffixs
SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.webp', '.pfm')


def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def sigmoid(x: ndarray) -> ndarray:
    return 1. / (1. + np.exp(-x))


def path_to_list(images_path: Union[str, Path]) -> List:
    if isinstance(images_path, str):
        images_path = Path(images_path)
    assert images_path.exists()
    if images_path.is_dir():
        images = [
            i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS
        ]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]
    return images


def crop_mask(masks: ndarray, bboxes: ndarray) -> ndarray:
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(bboxes[:, :, None], [1, 2, 3],
                              1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def box_iou(box1: ndarray, box2: ndarray) -> float:
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2
    x1 = max(x11, x12)
    y1 = max(y11, y12)
    x2 = min(x21, x22)
    y2 = min(y21, y22)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (x21 - x11) * (y21 - y11) + (x22 - x12) * (y22 - y12) - inter_area
    return max(0, inter_area / union_area)


def NMSBoxes(
        boxes: ndarray,
        scores: ndarray,
        labels: ndarray,
        iou_thres: float,
        agnostic: bool = False
):
    num_boxes = boxes.shape[0]
    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    labels = labels[order]

    indices = []

    for i in range(num_boxes):
        box_a = boxes[i]
        label_a = labels[i]
        keep = True
        for j in indices:
            box_b = boxes[j]
            label_b = labels[j]
            if not agnostic and label_a != label_b:
                continue
            if box_iou(box_a, box_b) > iou_thres:
                keep = False
        if keep:
            indices.append(i)

    indices = np.array(indices, dtype=np.int32)
    return order[indices]




def det_postprocess(data, conf_threshold=0.25, iou_threshold=0.65):
    """
    将 YOLOv8 TensorRT 原始输出 [1, 84, 8400] 解码为检测结果
    参数:
        data: 张量，形状 [1, 84, 8400]，其中:
            - 前 4 通道: [x, y, w, h] 中心坐标和宽高
            - 后 80 通道: 各类别置信度分数
        conf_threshold: 置信度阈值（默认 0.25）
        iou_threshold: NMS 的 IoU 阈值（默认 0.65）

    返回:
        bboxes: [N, 4] 左上右下坐标 (x1, y1, x2, y2)
        scores: [N] 各检测框的置信度
        labels: [N] 各检测框的类别 ID
    """

    # data: [1, 84, 8400] 或 [1, 85, 8400]
    if isinstance(data, (list, tuple)):
        # 兼容其他导出的情况，直接取第一个 Tensor
        data = data[0]
    assert isinstance(data, torch.Tensor), "det_postprocess: data 必须是 Tensor"

    # 第 1 步: 解析输入张量
    pred = data[0]  # [84, 8400]

    # 第 2 步: 拆分 bbox 和分类分数
    bbox_raw = pred[:4]  # [4, 8400]
    class_scores = pred[4:]  # [80, 8400]

    # 第 3 步: 获取每个点最高的类别分数和对应的类别 ID
    scores, labels = torch.max(class_scores, dim=0)  # 各返回 [8400]

    # 第 4 步: 按置信度过滤
    mask = scores >= conf_threshold
    scores = scores[mask]  # [M]
    labels = labels[mask]  # [M]
    bbox_raw = bbox_raw[:, mask]  # [4, M]

    if bbox_raw.shape[1] == 0:
        # 没有检测到目标，返回空张量
        device = data.device
        return (
            torch.zeros((0, 4), device=device),
            torch.zeros((0,), device=device),
            torch.zeros((0,), dtype=torch.long, device=device)
        )

    # 第 5 步: 将 [cx, cy, w, h] 转换为 [x1, y1, x2, y2]
    cx, cy, w, h = bbox_raw[0], bbox_raw[1], bbox_raw[2], bbox_raw[3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    bboxes = torch.stack([x1, y1, x2, y2], dim=1)  # [M, 4]

    # 第 6 步: 执行 NMS（非极大值抑制）
    keep_indices = torchvision.ops.nms(bboxes, scores, iou_threshold)

    # 第 7 步: 获取最终结果
    bboxes = bboxes[keep_indices]  # [N, 4]
    scores = scores[keep_indices]  # [N]
    labels = labels[keep_indices]  # [N]

    return bboxes, scores, labels


def seg_postprocess(
        data: Tuple[ndarray],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = (i[0] for i in data)
    bboxes, scores, labels, maskconf = np.split(outputs, [4, 5, 6], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return np.empty((0, 4), dtype=np.float32), \
            np.empty((0,), dtype=np.float32), \
            np.empty((0,), dtype=np.int32), \
            np.empty((0, 0, 0, 0), dtype=np.int32)

    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    cvbboxes = np.concatenate([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]],
                              1)
    labels = labels.astype(np.int32)
    v0, v1 = map(int, (cv2.__version__).split('.')[:2])
    assert v0 == 4, 'OpenCV version is wrong'
    if v1 > 6:
        idx = cv2.dnn.NMSBoxesBatched(cvbboxes, scores, labels, conf_thres,
                                      iou_thres)
    else:
        idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    masks = sigmoid(maskconf @ proto).reshape(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = masks.transpose([1, 2, 0])
    masks = cv2.resize(masks, (shape[1], shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    masks = masks.transpose(2, 0, 1)
    masks = np.ascontiguousarray((masks > 0.5)[..., None], dtype=np.float32)
    return bboxes, scores, labels, masks


def pose_postprocess(
        data: Union[Tuple, ndarray],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[ndarray, ndarray, ndarray]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = np.transpose(data[0], (1, 0))
    bboxes, scores, kpts = np.split(outputs, [4, 5], 1)
    scores, kpts = scores.squeeze(), kpts.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return np.empty((0, 4), dtype=np.float32), np.empty(
            (0,), dtype=np.float32), np.empty((0, 0, 0), dtype=np.float32)
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    xycenter, wh = np.split(bboxes, [
        2,
    ], -1)
    cvbboxes = np.concatenate([xycenter - 0.5 * wh, wh], -1)
    idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    cvbboxes, scores, kpts = cvbboxes[idx], scores[idx], kpts[idx]
    cvbboxes[:, 2:] += cvbboxes[:, :2]
    return cvbboxes, scores, kpts.reshape(idx.shape[0], -1, 3)
