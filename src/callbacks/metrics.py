import numpy as np

from catalyst.dl import MetricCallback, RunnerState


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
        iou_thresholds=(0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.iou_thresholds = np.array(iou_thresholds)

    def metric_value_from_ious(self, ious: np.ndarray):
        """

        Args:
            ious (np.ndarray): array of shape n_pred x n_gt
        Returns:

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
