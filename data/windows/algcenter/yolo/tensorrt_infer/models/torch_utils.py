from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms, nms
import torchvision

def seg_postprocess(
        data: Tuple[Tensor],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = data[0][0], data[1][0]
    bboxes, scores, labels, maskconf = outputs.split([4, 1, 1, 32], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), labels.new_zeros((0, )), bboxes.new_zeros((0, 0, 0, 0))
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    idx = batched_nms(bboxes, scores, labels, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx].int(), maskconf[idx]
    masks = (maskconf @ proto).sigmoid().view(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = F.interpolate(masks[None],
                          shape,
                          mode='bilinear',
                          align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    return bboxes, scores, labels, masks


def pose_postprocess(
        data: Union[Tuple, Tensor],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = torch.transpose(data[0], 0, 1).contiguous()
    bboxes, scores, kpts = outputs.split([4, 1, 51], 1)
    scores, kpts = scores.squeeze(), kpts.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), bboxes.new_zeros((0, 0, 0))
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    xycenter, wh = bboxes.chunk(2, -1)
    bboxes = torch.cat([xycenter - 0.5 * wh, xycenter + 0.5 * wh], -1)
    idx = nms(bboxes, scores, iou_thres)
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    return bboxes, scores, kpts.reshape(idx.shape[0], -1, 3)


# def det_postprocess(data: Tuple[Tensor, Tensor, Tensor, Tensor]):
#     assert len(data) == 4
#     iou_thres: float = 0.65
#     num_dets, bboxes, scores, labels = data[0][0], data[1][0], data[2][
#         0], data[3][0]
#     nums = num_dets.item()
#     if nums == 0:
#         return bboxes.new_zeros((0, 4)), scores.new_zeros(
#             (0, )), labels.new_zeros((0, ))
#     # check score negative
#     scores[scores < 0] = 1 + scores[scores < 0]
#     bboxes = bboxes[:nums]
#     scores = scores[:nums]
#     labels = labels[:nums]
#
#     return bboxes, scores, labels

def det_postprocess(data, conf_thres=0.25, iou_thres=0.45):
    """
    兼容单输出和多输出格式的后处理
    """
    import torch
    import torchvision

    # 调试信息（可选，成功后可删除）
    # print(f"DEBUG: data type = {type(data)}")

    # 单输出格式处理（标准 YOLOv8 格式）
    if isinstance(data, torch.Tensor):
        output = data
    elif isinstance(data, (list, tuple)) and len(data) == 1:
        output = data[0]
    else:
        # 多输出格式
        assert len(data) == 4, f"Expected 4 outputs, got {len(data)}"
        raise NotImplementedError("多输出格式处理 - 需要保留原代码逻辑")

    # 单输出格式解析
    # output shape: [batch, 84, 8400]
    batch_size = output.shape[0]

    # 转置: [batch, C, anchors] -> [batch, anchors, C]
    output = output.transpose(1, 2)  # [1, 8400, 84]

    boxes = output[..., :4]  # [1, 8400, 4] - cx, cy, w, h
    scores_all = output[..., 4:]  # [1, 8400, 80]

    # 获取最大分数和对应类别
    scores, labels = scores_all.max(dim=-1)  # [1, 8400]

    # 置信度过滤
    mask = scores > conf_thres

    # 只处理 batch=1 的情况（简化）
    box = boxes[0][mask[0]]  # [N, 4]
    score = scores[0][mask[0]]  # [N]
    label = labels[0][mask[0]]  # [N]

    if len(box) == 0:
        # 返回空 tensor
        return torch.zeros((0, 4), device=output.device), \
            torch.zeros(0, device=output.device), \
            torch.zeros(0, device=output.device, dtype=torch.long)

    # 转换 cx,cy,w,h -> x1,y1,x2,y2
    x1 = box[:, 0] - box[:, 2] / 2
    y1 = box[:, 1] - box[:, 3] / 2
    x2 = box[:, 0] + box[:, 2] / 2
    y2 = box[:, 1] + box[:, 3] / 2
    box_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    # NMS
    keep = torchvision.ops.nms(box_xyxy, score, iou_thres)

    return box_xyxy[keep], score[keep], label[keep]


def crop_mask(masks: Tensor, bboxes: Tensor) -> Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device,
                     dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device,
                     dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
