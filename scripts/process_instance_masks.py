from typing import List  # isort:skip
import argparse
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
from skimage import measure, morphology

from catalyst.utils import (
    get_pool,
    has_image_extension,
    imread,
    mimwrite_with_meta,
    tqdm_parallel_imap,
)


def build_args(parser):
    parser.add_argument(
        "--in-dir", type=Path, required=True, help="Raw masks folder path"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Processed masks folder path",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--n-channels",
        type=int,
        choices={2, 3},
        default=2,
        help="Number of channels in output masks",
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Number of workers to parallel the processing",
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def mim_interaction(mim: List[np.ndarray], threshold: float = 0) -> np.ndarray:
    result = np.zeros_like(mim[0], dtype=np.uint8)
    result[np.stack(mim, axis=-1).max(axis=-1) > threshold] = 255
    return result


def mim_color_encode(mim: List[np.ndarray], threshold: float = 0) -> np.ndarray:
    result = np.zeros_like(mim[0], dtype=np.uint8)
    for index, im in enumerate(mim, start=1):
        result[im > threshold] = index

    return result


class Preprocessor:
    def __init__(
        self,
        in_dir: Path,
        out_dir: Path,
        threshold: float = 0.0,
        n_channels: int = 2,
    ):
        """
        Args:
            in_dir (Path): raw masks folder path, input folder structure
                should be following:
                    in_path  # dir with raw masks
                    |-- sample_1
                    |   |-- instance_1
                    |   |-- instance_2
                    |   ..
                    |   `-- instance_N
                    |-- sample_1
                    |   |-- instance_1
                    |   |-- instance_2
                    |   ..
                    |   `-- instance_K
                    ..
                    `-- sample_M
                        |-- instance_1
                        |-- instance_2
                        ..
                        `-- instance_Z
            out_dir (Path): processed masks folder path, output folder
                structure will be following:
                    out_path
                        |-- sample_1.tiff  # image of shape HxWxN
                        |-- sample_2.tiff  # image of shape HxWxK
                        ..
                        `-- sample_M.tiff  # image of shape HxWxZ
            threshold (float):
            n_channels (int): number of channels in output masks,
                see https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741  # noqa: E501, W505
        """
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.threshold = threshold
        self.n_channels = n_channels

    def preprocess(self, sample: Path):
        masks = [
            imread(filename, grayscale=True, expand_dims=False)
            for filename in sample.iterdir()
            if has_image_extension(str(filename))
        ]
        labels = mim_color_encode(masks, self.threshold)

        scaled_blobs = morphology.dilation(labels > 0, morphology.square(9))
        watersheded_blobs = (
            morphology.watershed(
                scaled_blobs, labels, mask=scaled_blobs, watershed_line=True
            )
            > 0
        )
        watershed_lines = scaled_blobs ^ (watersheded_blobs)
        scaled_watershed_lines = morphology.dilation(
            watershed_lines, morphology.square(7)
        )

        props = measure.regionprops(labels)
        max_area = max(p.area for p in props)

        mask_without_borders = mim_interaction(masks, self.threshold)
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

                uniq = np.unique(
                    labels[
                        max(0, y0 - sz) : min(labels.shape[0], y0 + sz + 1),
                        max(0, x0 - sz) : min(labels.shape[1], x0 + sz + 1),
                    ]
                )
                if len(uniq[uniq > 0]) > 1:
                    borders[y0, x0] = 255
                    mask_without_borders[y0, x0] = 0

        if self.n_channels == 2:
            mask = [mask_without_borders, borders]
        elif self.n_channels == 3:
            background = 255 - (mask_without_borders + borders)
            mask = [mask_without_borders, borders, background]
        else:
            raise ValueError()

        mimwrite_with_meta(
            self.out_dir / f"{sample.stem}.tiff", mask, {"compress": 9}
        )

    def process_all(self, pool: Pool):
        images = list(self.in_dir.iterdir())
        tqdm_parallel_imap(self.preprocess, images, pool)


def main(args, _=None):
    args = args.__dict__
    args.pop("command", None)
    num_workers = args.pop("num_workers")

    with get_pool(num_workers) as p:
        Preprocessor(**args).process_all(p)


if __name__ == "__main__":
    args = parse_args()
    main(args)
