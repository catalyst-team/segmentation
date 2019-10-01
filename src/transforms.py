import cv2

import albumentations as A
from albumentations import (
    ChannelShuffle, CLAHE, Compose, HueSaturationValue, IAAPerspective,
    JpegCompression, LongestMaxSize, Normalize, OneOf, PadIfNeeded,
    RandomBrightnessContrast, RandomGamma, RGBShift, ShiftScaleRotate, ToGray
)

from catalyst.utils import tensor_from_rgb_image

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class ToTensor(A.core.transforms_interface.DualTransform):
    """Convert image and mask to ``torch.Tensor``"""

    def __call__(self, force_apply=True, **kwargs):
        kwargs.update(image=tensor_from_rgb_image(kwargs["image"]))
        if "mask" in kwargs.keys():
            kwargs.update(mask=tensor_from_rgb_image(kwargs["mask"]).float())

        return kwargs


def pre_transforms(image_size=256):
    transforms = Compose([
        LongestMaxSize(max_size=image_size),
        PadIfNeeded(
            image_size, image_size, border_mode=cv2.BORDER_CONSTANT
        ),
    ])
    return transforms


def post_transforms():
    return Compose([Normalize(), ToTensor()])


def hard_transform(image_size=256, p=0.5):
    transforms = Compose([
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=p,
        ),
        IAAPerspective(scale=(0.02, 0.05), p=p),
        OneOf([
            HueSaturationValue(p=p),
            ToGray(p=p),
            RGBShift(p=p),
            ChannelShuffle(p=p),
        ]),
        RandomBrightnessContrast(
            brightness_limit=0.5, contrast_limit=0.5, p=p
        ),
        RandomGamma(p=p),
        CLAHE(p=p),
        JpegCompression(quality_lower=50, p=p),
    ])
    return transforms
