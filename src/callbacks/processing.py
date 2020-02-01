import torch

from catalyst.dl import Callback, CallbackOrder, State
from .utils import encode_mask_with_color, label_instances


class RawMaskPostprocessingCallback(Callback):
    def __init__(
        self,
        threshold: float = 0.5,
        input_key: str = "logits",
        output_key: str = "mask",
    ):
        super().__init__(CallbackOrder.Internal)
        self.threshold = threshold
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state: State):
        output: torch.Tensor = torch.sigmoid(
            state.output[self.input_key].data.cpu()
        ).numpy()

        state.output[self.output_key] = \
            encode_mask_with_color(output, self.threshold)


class InstanceMaskPostprocessingCallback(Callback):
    def __init__(
        self,
        watershed_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        input_key: str = "logits",
        output_key: str = "instance_mask",
        out_key_semantic: str = None,
        out_key_border: str = None,
    ):
        super().__init__(CallbackOrder.Internal)
        self.watershed_threshold = watershed_threshold
        self.mask_threshold = mask_threshold
        self.input_key = input_key
        self.output_key = output_key
        self.out_key_semantic = out_key_semantic
        self.out_key_border = out_key_border

    def on_batch_end(self, state: State):
        output: torch.Tensor = torch.sigmoid(state.output[self.input_key])

        semantic, border = output.chunk(2, -3)

        if self.out_key_semantic is not None:
            state.output[self.out_key_semantic] = encode_mask_with_color(
                semantic.data.cpu().numpy(), threshold=self.mask_threshold
            )

        if self.out_key_border is not None:
            state.output[self.out_key_border] = (
                border.data.cpu().squeeze(-3).numpy() >
                self.watershed_threshold
            )

        state.output[self.output_key] = label_instances(
            semantic,
            border,
            watershed_threshold=self.watershed_threshold,
            instance_mask_threshold=self.mask_threshold,
            downscale_factor=1,
        )


__all__ = [
    "RawMaskPostprocessingCallback",
    "InstanceMaskPostprocessingCallback",
]
