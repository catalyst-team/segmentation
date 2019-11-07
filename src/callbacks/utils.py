from typing import List  # isort:skip

import numpy as np
from skimage.color import label2rgb

import torch


def encode_mask_with_color(
    semantic_masks: torch.Tensor, threshold: float = 0.5
) -> List[np.ndarray]:
    """

    Args:
        semantic_masks (torch.Tensor): semantic mask batch tensor
        threshold (float): threshold for semantic masks
    Returns:
        List[np.ndarray]: list of semantic masks

    """
    batch = []
    for observation in semantic_masks:
        result = np.zeros_like(observation[0], dtype=np.int32)
        for i, ch in enumerate(observation, start=1):
            result[ch > threshold] = i

        batch.append(result)

    return batch


def mask_to_overlay_image(
    image: np.ndarray, mask: np.ndarray, mask_strength: float
) -> np.ndarray:
    mask = label2rgb(mask, bg_label=0)
    image_with_overlay = image * (1 - mask_strength) + mask * mask_strength
    image_with_overlay = (
        (image_with_overlay * 255).clip(0, 255).round().astype(np.uint8)
    )
    return image_with_overlay
