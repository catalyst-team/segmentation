import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from .utils import encode_mask_with_color


class RawMaskPostprocessingCallback(Callback):
    """Callback to extract pixel masks from predicted logits."""

    def __init__(
        self,
        threshold: float = 0.5,
        input_key: str = "logits",
        output_key: str = "mask",
    ):
        """Constructor method for the :class:`RawMaskPostprocessingCallback`.

        Args:
            threshold (float): threshold for masks binarization
            input_key (str): input key to use for masks extraction
            output_key (str): key to use to store the result
        """
        super().__init__(order=CallbackOrder.Internal, node=CallbackNode.All)
        self.threshold = threshold
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, runner: IRunner):
        """Extract masks from predicted logits.

        Args:
            runner (IRunner): current runner
        """
        output = runner.output[self.input_key]

        output = torch.sigmoid(output).detach().cpu().numpy()
        runner.output[self.output_key] = encode_mask_with_color(
            output, self.threshold
        )


__all__ = ["RawMaskPostprocessingCallback"]
