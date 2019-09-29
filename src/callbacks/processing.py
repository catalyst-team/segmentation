import numpy as np

import torch

from catalyst.dl import Callback, CallbackOrder, RunnerState

from .utils import label_instances


class RawMaskPostprocessingCallback(Callback):
    def __init__(
        self,
        threshold: float = 0.5,
        input_key: str = "logits",
        output_key: str = "mask"
    ):
        super().__init__(CallbackOrder.Internal)
        self.threshold = threshold
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state: RunnerState):
        output: torch.Tensor = torch.sigmoid(
            state.output[self.input_key].data.cpu()
        ).numpy()

        batch = []
        for observation in output:
            result = np.zeros_like(observation[0], dtype=np.int32)
            for i, ch in enumerate(observation, start=1):
                result[ch >= self.threshold] = i

            batch.append(result)

        state.output[self.output_key] = batch


class InstanceMaskPostprocessingCallback(Callback):
    def __init__(
        self,
        watershed_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        input_key: str = "logits",
        output_key: str = "mask"
    ):
        super().__init__(CallbackOrder.Internal)
        self.watershed_threshold = watershed_threshold
        self.mask_threshold = mask_threshold
        self.input_key = input_key
        self.output_key = output_key

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
