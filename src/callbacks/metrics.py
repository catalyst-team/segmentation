import numpy as np

from catalyst.dl import MetricCallback


def compute_ious_single_image(predicted_mask, gt_instance_masks):
    instance_ids = np.unique(predicted_mask)
    n_gt_instaces = gt_instance_masks.shape[0]

    all_ious = []

    for id_ in instance_ids:
        if id_ == 0:
            # Skip background
            continue

        predicted_instance_mask = predicted_mask == id_

        sum_ = (predicted_instance_mask.reshape(1, -1)
                + gt_instance_masks.reshape(n_gt_instaces, -1))

        intersection = (sum_ == 2).sum(axis=1)
        union = (sum_ > 0).sum(axis=1)

        ious = intersection / union

        all_ious.append(ious)

    all_ious = np.array(all_ious).reshape((len(all_ious), n_gt_instaces))

    return all_ious


def map_from_ious(ious: np.ndarray, iou_thresholds: np.ndarray):
    """

    Args:
        ious (np.ndarray): array of shape n_pred x n_gt
    Returns:

    """
    n_preds = ious.shape[0]

    fn_at_ious = (
        np.max(ious, axis=0, initial=0)[None, :] < iou_thresholds[:, None]
    )
    fn_at_iou = np.sum(fn_at_ious, axis=1, initial=0)

    tp_at_ious = (
        np.max(ious, axis=0, initial=0)[None, :] > iou_thresholds[:, None]
    )
    tp_at_iou = np.sum(tp_at_ious, axis=1, initial=0)

    metric_at_iou = tp_at_iou / (n_preds + fn_at_iou)

    return metric_at_iou.mean()


def mean_average_precision(outputs, targets, iou_thresholds):
    batch_metrics = []
    for pred, gt in zip(outputs, targets):
        ious = compute_ious_single_image(pred, gt.numpy())
        batch_metrics.append(map_from_ious(ious, iou_thresholds))
    return float(np.mean(batch_metrics))


class SegmentationMeanAPCallback(MetricCallback):
    def __init__(
        self,
        input_key: str = "imasks",
        output_key: str = "instance_mask",
        prefix: str = "mAP",
        iou_thresholds=(0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
    ):
        super().__init__(
            prefix=prefix,
            metric_fn=mean_average_precision,
            input_key=input_key,
            output_key=output_key,
            iou_thresholds=np.array(iou_thresholds),
        )
