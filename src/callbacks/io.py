from pathlib import Path

import imageio
import numpy as np

from catalyst.dl import Callback, CallbackOrder, State, utils
from .utils import crop_by_masks, mask_to_overlay_image


class OriginalImageSaverCallback(Callback):
    def __init__(
        self,
        output_dir: str,
        relative: bool = True,
        filename_suffix: str = "",
        filename_extension: str = ".jpg",
        input_key: str = "image",
        outpath_key: str = "name",
    ):
        super().__init__(CallbackOrder.Logging)
        self.output_dir = Path(output_dir)
        self.relative = relative
        self.filename_suffix = filename_suffix
        self.filename_extension = filename_extension
        self.input_key = input_key
        self.outpath_key = outpath_key

    def get_image_path(self, state: State, name: str, suffix: str = ""):
        out_dir = (
            state.logdir / self.output_dir
            if self.relative
            else self.output_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        res = out_dir / f"{name}{suffix}{self.filename_extension}"

        return res

    def on_batch_end(self, state: State):
        names = state.batch_in[self.outpath_key]
        images = state.batch_in[self.input_key]

        images = utils.tensor_to_ndimage(images.detach().cpu(), dtype=np.uint8)
        for image, name in zip(images, names):
            fname = self.get_image_path(state, name, self.filename_suffix)
            imageio.imwrite(fname, image)


class OverlayMaskImageSaverCallback(OriginalImageSaverCallback):
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

    def on_batch_end(self, state: State):
        names = state.batch_in[self.outpath_key]
        images = state.batch_in[self.input_key]
        masks = state.batch_out[self.output_key]

        images = utils.tensor_to_ndimage(images.detach().cpu())
        for name, image, mask in zip(names, images, masks):
            image = mask_to_overlay_image(image, mask, self.mask_strength)
            fname = self.get_image_path(state, name, self.filename_suffix)
            imageio.imwrite(fname, image)


class InstanceCropSaverCallback(OriginalImageSaverCallback):
    def __init__(
        self,
        output_dir: str,
        relative: bool = True,
        filename_extension: str = ".jpg",
        input_key: str = "image",
        output_key: str = "mask",
        outpath_key: str = "name",
    ):
        super().__init__(
            output_dir=output_dir,
            relative=relative,
            filename_suffix=filename_extension,
            input_key=input_key,
            outpath_key=outpath_key,
        )
        self.output_key = output_key

    def on_batch_end(self, state: State):
        names = state.input[self.outpath_key]
        images = utils.tensor_to_ndimage(state.input[self.input_key].cpu())
        masks = state.output[self.output_key]

        for name, image, masks_ in zip(names, images, masks):
            instances = crop_by_masks(image, masks_)

            for index, crop in enumerate(instances):
                filename = self.get_image_path(
                    state, name, suffix=f"_instance{index:02d}"
                )
                imageio.imwrite(filename, crop)


__all__ = [
    "OriginalImageSaverCallback",
    "OverlayMaskImageSaverCallback",
    "InstanceCropSaverCallback",
]
