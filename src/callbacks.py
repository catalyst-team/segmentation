from pathlib import Path
import numpy as np

import imageio
from skimage.color import label2rgb

import torch

from catalyst.dl import Callback, MetricCallback, RunnerState
from catalyst.utils.image import tensor_to_ndimage

from .utils import crop_by_masks, label_instances


# -----------------------------------------------------------------------------
# processing


class RawMaskPostprocessing(Callback):
    def __init__(
        self,
        input_key: str = "logits",
        output_key: str = "mask",
        threshold: float = 0.5,
    ):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.threshold = threshold

    def on_batch_end(self, state: RunnerState):
        output: torch.Tensor = torch.sigmoid(
            state.output[self.input_key].cpu()
        ).numpy()

        batch = []
        for observation in output:
            result = np.zeros_like(observation[0], dtype=np.int32)
            for i, ch in enumerate(observation):
                result[ch >= self.threshold] = i + 1

            batch.append(result)

        state.output[self.output_key] = batch


class InstanceMaskCallback(Callback):
    def __init__(
        self,
        input_key: str = "logits",
        output_key: str = "mask",
        watershed_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.watershed_threshold = watershed_threshold
        self.mask_threshold = mask_threshold

    def on_batch_end(self, state: RunnerState):
        output: torch.Tensor = torch.sigmoid(state.output[self.input_key])

        semantic, border = output.chunk(2, -3)

        state.output[self.output_key] = label_instances(
            semantic,
            border,
            self.watershed_threshold,
            self.mask_threshold,
            downscale_factor=1,
        )


# -----------------------------------------------------------------------------
# cropping


class InstanceCropCallback(Callback):
    def __init__(self, output_dir: str, relative=True):
        self.relative = relative
        self.output_dir = output_dir

    def get_image_path(self, name, no, state: RunnerState):
        if self.relative:
            out_dir = state.logdir / self.output_dir
        else:
            out_dir = Path(self.output_dir)

        out_dir.mkdir(exist_ok=True)

        res = out_dir / f"{name}_03_instance{no:02d}.jpg"

        return res

    def on_batch_end(self, state):
        original = state.input["image"].cpu()
        original = tensor_to_ndimage(original, dtype=np.uint8)
        names = state.input["name"]

        for image, name, masks in zip(
            original, names, state.output["instance_mask"]
        ):
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
        filename_suffix=".jpg",
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

            image = (
                image * (1 - self.mask_strength) + mask * self.mask_strength
            )
            image = (image * 255).clip(0, 255).round().astype(np.uint8)

            imageio.imwrite(self.get_image_path(name, state), image)


class OriginalImageSaverCallback(Callback):
    """Generic callback to save input images during inference"""

    def __init__(
        self,
        output_dir: str,
        relative=True,
        input_key="image",
        filename_suffix=".jpg",
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


class InstanceSegmentationMeanAPCallback(MetricCallback):
    @staticmethod
    def compute_ious_single_image(predicted_mask, gt_instance_masks):
        instance_ids = np.unique(predicted_mask)
        n_gt_instaces = gt_instance_masks.shape[0]

        all_ious = []

        for id_ in instance_ids:
            if id_ == 0:
                # Skip background
                continue

            predicted_instance_mask = predicted_mask == id_

            sum_ = predicted_instance_mask.reshape(
                1, -1
            ) + gt_instance_masks.reshape(n_gt_instaces, -1)

            intersection = (sum_ == 2).sum(axis=1)
            union = (sum_ > 0).sum(axis=1)

            ious = intersection / union

            all_ious.append(ious)

        all_ious = np.array(all_ious).reshape((len(all_ious), n_gt_instaces))

        return all_ious

    def __init__(
        self,
        input_key: str = "imasks",
        output_key: str = "instance_mask",
        iou_thresholds=(0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95),
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
            < self.iou_thresholds[:, None]
        )
        fn_at_iou = np.sum(fn_at_ious, axis=1, initial=0)

        tp_at_ious = (
            np.max(ious, axis=0, initial=0)[None, :]
            > self.iou_thresholds[:, None]
        )
        tp_at_iou = np.sum(tp_at_ious, axis=1, initial=0)

        metric_at_iou = tp_at_iou / (n_preds + fn_at_iou)

        return metric_at_iou.mean()

    def on_batch_end(self, state: RunnerState):
        merged_instance_mask = state.output[self.output_key]
        gt_instace_masks = state.input[self.input_key]

        batch_metrics = []
        for pred, gt in zip(merged_instance_mask, gt_instace_masks):
            ious = self.compute_ious_single_image(pred, gt.numpy())
            batch_metrics.append(self.metric_value_from_ious(ious))

        state.batch_metrics["mAP"] = float(np.mean(batch_metrics))
