from typing import List, Tuple, Union
import numpy as np

import cv2
from shapely.geometry import LinearRing, MultiPoint
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import watershed

import torch
import torch.nn.functional as F


# types
Point = Tuple[int, int]
Quadrangle = Tuple[Point, Point, Point, Point]


def encode_mask_with_color(
    semantic_masks: torch.Tensor, threshold: float = 0.5
) -> List[np.ndarray]:
    """

    Args:
        semantic_masks: semantic mask batch tensor
        threshold: threshold for semantic masks
    Returns:
        List of semantic masks

    """
    batch = []
    for observation in semantic_masks:
        result = np.zeros_like(observation[0], dtype=np.int32)
        for i, ch in enumerate(observation, start=1):
            result[ch > threshold] = i

        batch.append(result)

    return batch


def label_instances(
    semantic_masks: torch.Tensor,
    border_masks: torch.Tensor,
    watershed_threshold: float = 0.9,
    instance_mask_threshold: float = 0.5,
    downscale_factor: float = 4,
    interpolation: str = "bilinear"
) -> List[np.ndarray]:
    """

    Args:
        semantic_masks: semantic mask batch tensor
        border_masks:  instance mask batch tensor
        watershed_threshold: threshold for watershed markers
        instance_mask_threshold: threshold for final instance masks
        downscale_factor: mask downscaling factor (to speed up processing)
        interpolation: interpolation method
    Returns:
        List of labeled instance masks, one per batch item

    """
    bordered_masks = (semantic_masks - border_masks).clamp(min=0)

    scaling = 1 / downscale_factor
    semantic_masks, bordered_masks = (
        F.interpolate(
            mask.data.cpu(),
            scale_factor=scaling,
            mode=interpolation,
            align_corners=False,
        )
        .squeeze(-3)
        .numpy()
        for mask in (semantic_masks, bordered_masks)
    )

    result: List[np.ndarray] = []
    for semantic, bordered in zip(semantic_masks, bordered_masks):
        watershed_marks = label(bordered > watershed_threshold, background=0)
        instance_regions = watershed(-bordered, watershed_marks)

        instance_regions[semantic < instance_mask_threshold] = 0

        result.append(instance_regions)

    return result


def _is_ccw(vertices: np.ndarray):
    return LinearRing(vertices * [[1, -1]]).is_ccw


def get_rects_from_mask(
    label_mask: np.ndarray, min_area_fraction=20
) -> np.ndarray:
    props = regionprops(label_mask)

    total_h, total_w = label_mask.shape
    total_area = total_h * total_w

    result = []
    for p in props:

        if p.area / total_area < min_area_fraction:
            continue

        coords = p.coords
        coords = coords[:, ::-1]  # row, col -> col, row

        rect = MultiPoint(coords).minimum_rotated_rectangle.exterior.coords

        rect = np.array(rect)[:4].astype(np.int32)

        if _is_ccw(rect):
            rect = rect[::-1]

        result.append(rect)

    result = np.stack(result) if result else []

    return result


def perspective_crop(
    image: np.ndarray,
    crop_coords: Union[Quadrangle, np.ndarray],
    output_wh: Tuple[int, int],
    border_color=(255, 255, 255)
):
    width, height = output_wh
    target_coords = ((0, 0), (width, 0), (width, height), (0, height))

    transform_matrix = cv2.getPerspectiveTransform(
        np.array(crop_coords, dtype=np.float32),
        np.array(target_coords, dtype=np.float32),
    )

    result = cv2.warpPerspective(
        image,
        transform_matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_color),
    )

    return result


def perspective_crop_keep_ratio(
    image: np.ndarray,
    vertices: np.ndarray,
    output_size: int = -1,
    border_color=(255, 255, 255)
) -> np.ndarray:
    """
    Crop some quadrilateral from image keeping it's aspect ratio
    Args:
        image: image numpy array
        vertices: numpy array with quadrilateral vertices coords
        output_size: minimal side length of output image (if -1 will be
            actual side of image)
        border_color:
    Returns:

    """
    lenghts = np.linalg.norm(vertices - np.roll(vertices, -1, 0), axis=1)

    len_ab, len_bc, len_cd, len_da = lenghts.tolist()

    width = (len_ab + len_cd) / 2
    height = (len_bc + len_da) / 2

    if output_size > 0:
        scale = output_size / max(width, height)
        width, height = (dim * scale for dim in (width, height))

    width, height = round(width), round(height)

    crop = perspective_crop(image, vertices, (width, height), border_color)

    return crop


def crop_by_masks(
    image: np.ndarray, mask: np.ndarray, image_size: int = 512
) -> List[np.ndarray]:
    crops = [
        perspective_crop_keep_ratio(image, rect, image_size)
        for rect in get_rects_from_mask(mask)
    ]

    return crops


def mask_to_overlay_image(
    image: np.ndarray, mask: np.ndarray, mask_strength: float
) -> np.ndarray:
    mask = label2rgb(mask, bg_label=0)
    image_with_overlay = image * (1 - mask_strength) + mask * mask_strength
    image_with_overlay = (
        (image_with_overlay * 255).clip(0, 255).round().astype(np.uint8)
    )
    return image_with_overlay
