from pathlib import Path

import imageio
import numpy as np

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.dl import utils
from .utils import mask_to_overlay_image


class OriginalImageSaverCallback(Callback):
    """Callback to save augmented images."""

    def __init__(
        self,
        output_dir: str,
        relative: bool = True,
        filename_suffix: str = "",
        filename_extension: str = ".jpg",
        input_key: str = "image",
        outpath_key: str = "name",
    ):
        """Constructor method for the :class:`OriginalImagesSaverCallback`.

        Args:
            output_dir (str): path to folder to store images
            relative (bool): flag to define whether `output_dir` is the path
                relative to the experiment logdir or not
            filename_suffix (str): suffix to add to the image names
            filename_extension (str): image file format
            input_key (str): key to use to get images
            outpath_key (str): key to use to get images names
        """
        super().__init__(order=CallbackOrder.logging, node=CallbackNode.master)
        self.output_dir = Path(output_dir)
        self.relative = relative
        self.suffix = filename_suffix
        self.extension = filename_extension
        self.input_key = input_key
        self.outpath_key = outpath_key

    def _get_image_path(self, logdir: Path, name: str):
        out_dir = (
            logdir / self.output_dir if self.relative else self.output_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        res = out_dir / f"{name}{self.suffix}{self.extension}"

        return res

    def on_batch_end(self, runner: IRunner):
        """Save batch of images.

        Args:
            runner (IRunner): current runner
        """
        names = runner.input[self.outpath_key]
        images = runner.input[self.input_key]

        images = utils.tensor_to_ndimage(images.detach().cpu(), dtype=np.uint8)
        for image, name in zip(images, names):
            fname = self._get_image_path(runner.logdir, name)
            imageio.imwrite(fname, image)


class OverlayMaskImageSaverCallback(OriginalImageSaverCallback):
    """Callback to draw predicted masks over images and save them."""

    def __init__(
        self,
        output_dir: str,
        relative: bool = True,
        mask_strength: float = 0.5,
        filename_suffix: str = "",
        filename_extension: str = ".jpg",
        input_key: str = "image",
        output_key: str = "mask",
        outpath_key: str = "name",
    ):
        """Constructor method for the :class:`OverlayMaskImageSaverCallback`.

        Args:
            output_dir (str): path to folder to store images
            relative (bool): flag to define whether `output_dir` is the path
                relative to the experiment logdir or not
            mask_strength (float): opacity of colorized masks
            filename_suffix (str): suffix to add to the image names
            filename_extension (str): image file format
            input_key (str): key to use to get images
            output_key (str): key to use to get predicted masks
            outpath_key (str): key to use to get images names
        """
        super().__init__(
            output_dir=output_dir,
            relative=relative,
            filename_suffix=filename_suffix,
            filename_extension=filename_extension,
            input_key=input_key,
            outpath_key=outpath_key,
        )
        self.mask_strength = mask_strength
        self.output_key = output_key

    def on_batch_end(self, runner: IRunner):
        """Save batch of images with overlay.

        Args:
            runner (IRunner): current runner
        """
        names = runner.input[self.outpath_key]
        images = runner.input[self.input_key]
        masks = runner.output[self.output_key]

        images = utils.tensor_to_ndimage(images.detach().cpu())
        for name, image, mask in zip(names, images, masks):
            image = mask_to_overlay_image(image, mask, self.mask_strength)
            fname = self._get_image_path(runner.logdir, name)
            imageio.imwrite(fname, image)


__all__ = ["OriginalImageSaverCallback", "OverlayMaskImageSaverCallback"]
