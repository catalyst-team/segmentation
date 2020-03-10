import torch

from catalyst.dl import Callback, CallbackNode, CallbackOrder, State
from .utils import encode_mask_with_color, label_instances


class RawMaskPostprocessingCallback(Callback):
    def __init__(
        self,
        threshold: float = 0.5,
        input_key: str = "logits",
        output_key: str = "mask",
    ):
        super().__init__(order=CallbackOrder.Internal, node=CallbackNode.All)
        self.threshold = threshold
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state: State):
        output = state.batch_out[self.input_key]

        output = torch.sigmoid(output).detach().cpu().numpy()
        state.batch_out[self.output_key] = encode_mask_with_color(
            output, threshold=self.threshold
        )


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
        super().__init__(CallbackOrder.Internal, node=CallbackNode.All)
        self.watershed_threshold = watershed_threshold
        self.mask_threshold = mask_threshold
        self.input_key = input_key
        self.output_key = output_key
        self.out_key_semantic = out_key_semantic
        self.out_key_border = out_key_border

    def on_batch_end(self, state: State):
        output = state.batch_out[self.input_key]

        output = torch.sigmoid(output).detach().cpu()
        semantic, border = output.chunk(2, -3)

        if self.out_key_semantic is not None:
            state.batch_out[self.out_key_semantic] = encode_mask_with_color(
                semantic.numpy(), threshold=self.mask_threshold
            )

        if self.out_key_border is not None:
            state.batch_out[self.out_key_border] = (
                border.squeeze(-3).numpy() > self.watershed_threshold
            )

        state.batch_out[self.output_key] = label_instances(
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
