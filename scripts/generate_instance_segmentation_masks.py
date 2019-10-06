from typing import List
import argparse
import os
import shutil
import numpy as np
from skimage import measure, morphology

from tqdm import tqdm

from catalyst.utils import boolean_flag, imread, mimwrite_with_meta

from.utils import find_images_in_dir


def build_args(parser):
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--n-channels", type=int, choices={2, 3}, default=2)
    boolean_flag(parser, "verbose", default=False)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def mimread_from_dir(dirpath: str, **kwargs) -> List[np.ndarray]:
    rootpath = kwargs.pop("rootpath", None)
    if rootpath is not None and not dirpath.startswith(rootpath):
        dirpath = os.path.join(rootpath, dirpath)

    mim = [
        imread(fname, rootpath=dirpath, **kwargs).astype(np.uint8)
        for fname in find_images_in_dir(dirpath)
    ]
    return mim


def mim_interaction(mim: List[np.ndarray], threshold: float = 0) -> np.ndarray:
    result = np.zeros_like(mim[0], dtype=np.uint8)
    result[np.stack(mim, axis=-1).max(axis=-1) > threshold] = 255
    return result


def mim_color_encode(
    mim: List[np.ndarray], threshold: float = 0
) -> np.ndarray:
    result = np.zeros_like(mim[0], dtype=np.uint8)
    for index, im in enumerate(mim, start=1):
        result[im > threshold] = index

    return result


def main(args, _=None):
    """

    in_path
    |-- sample_1
    |   |-- images
    |   |   `-- sample_1
    |   `-- masks
    |       |-- instance_1
    |       ..
    |       `-- instance_M
    ..
    `-- sample_N
        |-- images
        |   `-- sample_N
        `-- masks
            |-- instance_1
            ..
            `-- instance_K

    out_path
    |-- images
    |   |-- sample_1
    |   ..
    |   `-- sample_N
    `-- masks
        |-- sample_1.tiff  # image of shape HxWxM
        ..
        `-- sample_N.tiff  # image of shape HxWxK

    https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741

    """
    for sample in tqdm(os.listdir(args.in_dir), disable=(not args.verbose)):
        masks = mimread_from_dir(
            "masks",
            rootpath=f"{args.in_dir}/{sample}",
            grayscale=True,
            expand_dims=False
        )

        labels = mim_color_encode(masks, args.threshold)

        scaled_blobs = morphology.dilation(labels > 0, morphology.square(9))
        watersheded_blobs = morphology.watershed(
            scaled_blobs, labels, mask=scaled_blobs, watershed_line=True
        ) > 0
        watershed_lines = scaled_blobs ^ (watersheded_blobs)
        scaled_watershed_lines = morphology.dilation(
            watershed_lines, morphology.square(7)
        )

        props = measure.regionprops(labels)
        max_area = max(p.area for p in props)

        mask_without_borders = mim_interaction(masks, args.threshold)
        borders = np.zeros_like(labels, dtype=np.uint8)
        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not scaled_watershed_lines[y0, x0]:
                    continue

                if labels[y0, x0] == 0:
                    if max_area > 4000:
                        sz = 6
                    else:
                        sz = 3
                else:
                    if props[labels[y0, x0] - 1].area < 300:
                        sz = 1
                    elif props[labels[y0, x0] - 1].area < 2000:
                        sz = 2
                    else:
                        sz = 3

                uniq = np.unique(labels[
                    max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                    max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)
                ])
                if len(uniq[uniq > 0]) > 1:
                    borders[y0, x0] = 255
                    mask_without_borders[y0, x0] = 0

        os.makedirs(f"{args.out_dir}/images", exist_ok=True)
        image = find_images_in_dir(f"{args.in_dir}/{sample}/images")[0]
        shutil.copy(image, f"{args.out_dir}/images/{os.path.basename(image)}")

        os.makedirs(f"{args.out_dir}/masks", exist_ok=True)
        if args.n_channels == 2:
            mask = [mask_without_borders, borders]
        elif args.n_channels == 3:
            background = 255 - (mask_without_borders + borders)
            mask = [mask_without_borders, borders, background]
        else:
            raise ValueError()
        mimwrite_with_meta(
            f"{args.out_dir}/masks/{sample}.tiff", mask, {"compress": 9}
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
