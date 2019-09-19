from typing import Union, Tuple, List
from pathlib import Path
import numpy as np

import cv2
import imageio
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import watershed
from shapely.geometry import MultiPoint, LinearRing

import torch
import torch.nn.functional as F

from catalyst.dl import Callback, RunnerState
from catalyst.utils.image import tensor_to_ndimage


# -----------------------------------------------------------------------------
# types

Point = Tuple[int, int]
Quadrangle = Tuple[Point, Point, Point, Point]


# -----------------------------------------------------------------------------
# utils


def label_instances(
    semantic_masks: torch.Tensor,
    border_masks: torch.Tensor,
    watershed_threshold=.9,
    instance_mask_threshold=.5,
    downscale_factor=4,
) -> List[np.ndarray]:
    """

    Args:
        semantic_masks: semantic mask batch tensor
        border_masks:  instance mask batch tensor
        watershed_threshold: threshold for watershed markers
        instance_mask_threshold: threshold for final instance masks
        downscale_factor: mask downscaling factor (to speed up processing)
    Returns:
        List of labeled instance masks, one per batch item
    """

    bordered_masks = (semantic_masks - border_masks).clamp(min=0)

    scaling = 1 / downscale_factor
    semantic_masks = F.interpolate(
        semantic_masks,
        scale_factor=scaling,
        mode="bilinear",
        align_corners=False
    ).cpu().squeeze(-3).numpy()

    bordered_masks = F.interpolate(
        bordered_masks,
        scale_factor=scaling,
        mode="bilinear",
        align_corners=False
    ).cpu().squeeze(-3).numpy()

    res: List[np.ndarray] = []
    for semantic, bordered in zip(semantic_masks, bordered_masks):
        watershed_marks = label(bordered > watershed_threshold, background=0)
        instance_regions = watershed(-bordered, watershed_marks)

        instance_regions[semantic < instance_mask_threshold] = 0

        res.append(instance_regions)

    return res


def _is_ccw(vertices: np.ndarray):
    return LinearRing(vertices * [[1, -1]]).is_ccw


def get_rects_from_mask(
    label_mask: np.ndarray,
    min_area_fraction=20
) -> np.ndarray:
    props = regionprops(label_mask)

    total_h, total_w = label_mask.shape
    total_area = total_h * total_w

    res = []
    for p in props:

        if p.area / total_area < min_area_fraction:
            continue

        coords = p.coords
        coords = coords[:, ::-1]  # row, col -> col, row

        rect = MultiPoint(coords).minimum_rotated_rectangle.exterior.coords

        rect = np.array(rect)[:4].astype(np.int32)

        if _is_ccw(rect):
            rect = rect[::-1]

        res.append(rect)

    res = np.stack(res)

    return res


def perspective_crop(
    image: np.ndarray,
    crop_coords: Union[Quadrangle, np.ndarray],
    output_wh: Tuple[int, int],
    border_color=(255, 255, 255)
):
    w, h = output_wh
    target_coords = (
        (0, 0), (w, 0),
        (w, h), (0, h)
    )

    transform_matrix = cv2.getPerspectiveTransform(
        np.array(crop_coords, dtype=np.float32),
        np.array(target_coords, dtype=np.float32)
    )

    res = cv2.warpPerspective(
        image, transform_matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_color
    )

    return res


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

    w = (len_ab + len_cd) / 2
    h = (len_bc + len_da) / 2

    if output_size > 0:
        if h > w:
            scale = output_size / h
            h, w = output_size, w * scale
        else:
            scale = output_size / w
            h, w = h * scale, output_size

    h, w = round(h), round(w)

    crop = perspective_crop(image, vertices, (w, h), border_color)

    return crop


def crop_by_masks(image: np.ndarray, mask: np.ndarray, image_size: int = 512):
    crops = []

    for rect in get_rects_from_mask(mask):
        crops.append(perspective_crop_keep_ratio(image, rect, image_size))

    return crops


# -----------------------------------------------------------------------------
# processing


class RawMaskPostprocessing(Callback):
    def __init__(self, threshold=.5):
        super().__init__()
        self.threshold = threshold

    def on_batch_end(self, state: RunnerState):
        output: torch.Tensor = torch.sigmoid(
            state.output["logits"].cpu()
        ).numpy()

        batch = []
        for observation in output:

            result = np.zeros_like(observation[0], dtype=np.int32)

            for i, ch in enumerate(observation):
                result[ch >= self.threshold] = i + 1

            batch.append(result)

        state.output["mask"] = batch


class InstanceMaskCallback(Callback):

    def __init__(
        self,
        watershed_threshold: float = .5,
        mask_threshold: float = .5
    ):
        super().__init__()
        self.watershed_threshold = watershed_threshold
        self.mask_threshold = mask_threshold

    def on_batch_end(self, state: RunnerState):
        output: torch.Tensor = torch.sigmoid(state.output["logits"])

        semantic, border = output.chunk(2, -3)

        state.output["instance_mask"] = \
            label_instances(
                semantic,
                border,
                self.watershed_threshold,
                self.mask_threshold,
                downscale_factor=1
            )

# -----------------------------------------------------------------------------
# cropping


class InstanceCropCallback(Callback):
    def __init__(
        self,
        output_dir: str,
        relative=True
    ):
        self.relative = relative
        self.output_dir = output_dir

    def get_image_path(self, name, no, state: RunnerState):
        if self.relative:
            out_dir = state.logdir / self.output_dir
        else:
            out_dir = Path(self.output_dir)

        out_dir.mkdir(exist_ok=True)

        res = out_dir / f'{name}_03_instance{no:02d}.jpg'

        return res

    def on_batch_end(self, state):
        original = state.input["image"].cpu()
        original = tensor_to_ndimage(original, dtype=np.uint8)
        names = state.input['name']

        for image, name, masks in \
            zip(original, names, state.output["instance_mask"]):
            instances = crop_by_masks(image, masks)

            for no, crop in enumerate(instances):
                imageio.imwrite(self.get_image_path(name, no, state), crop)

# -----------------------------------------------------------------------------
# io


class OverlayMaskImageSaverCallback(Callback):
    def __init__(
        self,
        images_dir: str,
        relative=True,
        mask_strength=0.5,
        output_key="mask",
        filename_suffix=".jpg"
    ):
        self.relative = relative
        self.filename_suffix = filename_suffix
        self.output_key = output_key
        self.mask_strength = mask_strength
        self.output_dir = Path(images_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)

    def get_image_path(self, name, state: RunnerState):
        if self.relative:
            out_dir = state.logdir / self.output_dir
        else:
            out_dir = Path(self.output_dir)

        out_dir.mkdir(exist_ok=True)

        res = out_dir / f"{name}{self.filename_suffix}"

        return res

    def on_batch_end(self, state: RunnerState):
        names = state.input["name"]
        images = state.input["image"].cpu()
        masks = state.output[self.output_key]

        images = tensor_to_ndimage(images)

        for name, image, mask in zip(names, images, masks):
            mask = label2rgb(mask, bg_label=0)

            image = image * (1 - self.mask_strength) + mask * self.mask_strength
            image = (image * 255).clip(0, 255).round().astype(np.uint8)

            imageio.imwrite(self.get_image_path(name, state), image)


class OriginalImageSaverCallback(Callback):
    """Generic callback to save input images during inference"""
    def __init__(
        self,
        output_dir: str,
        relative=True,
        input_key="image",
        filename_suffix=".jpg"
    ):
        self.relative = relative
        self.output_dir = output_dir
        self.filename_suffix = filename_suffix
        self.input_key = input_key

    def get_image_path(self, name, state: RunnerState):
        if self.relative:
            out_dir = Path(state.logdir) / self.output_dir
        else:
            out_dir = Path(self.output_dir)

        out_dir.mkdir(exist_ok=True)

        res = out_dir / f"{name}{self.filename_suffix}"

        return res

    def on_batch_end(self, state):
        images = state.input[self.input_key].cpu()
        images = tensor_to_ndimage(images, dtype=np.uint8)
        names = state.input["name"]

        for image, name in zip(images, names):
            imageio.imwrite(self.get_image_path(name, state), image)

# -----------------------------------------------------------------------------
# metrics


class InstanceSegmentationMeanAPCallback(Callback):

    @staticmethod
    def compute_ious_single_image(predicted_mask, gt_instance_masks):
        instance_ids = np.unique(predicted_mask)
        n_gt_instaces = gt_instance_masks.shape[0]

        all_ious = []

        for id in instance_ids:
            if id == 0:
                # Skip background
                continue

            predicted_instance_mask = predicted_mask == id

            sum = \
                predicted_instance_mask.reshape(1, -1) + \
                gt_instance_masks.reshape(n_gt_instaces, -1)

            intersection = (sum == 2).sum(axis=1)
            union = (sum > 0).sum(axis=1)

            ious = intersection / union

            all_ious.append(ious)

        all_ious = np.array(all_ious).reshape((len(all_ious), n_gt_instaces))

        return all_ious

    def __init__(
        self,
        input_key="imasks",
        output_key="instance_mask",
        iou_thresholds=(.5, .55, .6, .7, .75, .8, .9, .95)
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.iou_thresholds = np.array(iou_thresholds)

    def metric_value_from_ious(self, ious: np.ndarray):
        """
        :param ious: np.ndarray (n_pred x n_gt)
        :return:
        """
        n_preds = ious.shape[0]

        fn_at_ious = (
                np.max(ious, axis=0, initial=0)[None, :]
                < self.iou_thresholds[:, None])
        fn_at_iou = np.sum(fn_at_ious, axis=1, initial=0)

        tp_at_ious = (
                np.max(ious, axis=0, initial=0)[None, :]
                > self.iou_thresholds[:, None])
        tp_at_iou = np.sum(tp_at_ious, axis=1, initial=0)

        metric_at_iou = tp_at_iou / (n_preds + fn_at_iou)

        return metric_at_iou.mean()

    def on_batch_end(self, state: RunnerState):
        merged_instance_mask = state.output[self.output_key]
        gt_instace_masks = state.input[self.input_key]

        batch_metrics = []
        for pred, gt in zip(merged_instance_mask, gt_instace_masks):
            ious = \
                InstanceSegmentationMeanAPCallback.compute_ious_single_image(
                    pred, gt.numpy())

            batch_metrics.append(self.metric_value_from_ious(ious))

        state.batch_metrics["mAP"] = float(np.mean(batch_metrics))
