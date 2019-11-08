from typing import Set  # isort:skip
import argparse
import collections
import functools
import json
import os

import numpy as np

from catalyst.utils import get_pool, imread, tqdm_parallel_imap


def build_args(parser):
    parser.add_argument(
        "--in-dir",
        required=True,
        type=str,
        help="Path to directory with dataset"
    )
    parser.add_argument(
        "--out-labeling",
        required=True,
        type=str,
        help="Path to output JSON"
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Number of workers to parallel the processing"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def colors_in_image(uri) -> Set:
    image = imread(uri, rootpath=args.in_dir)
    colors = np.unique(image.reshape(-1, image.shape[-1]), axis=0)
    result = {tuple(row) for row in colors.tolist()}  # np.array to hashable
    return result


def main(args, _=None):
    with get_pool(args.num_workers) as pool:
        images = os.listdir(args.in_dir)
        colors = tqdm_parallel_imap(colors_in_image, images, pool)
        unique_colors = functools.reduce(lambda s1, s2: s1 | s2, colors)

    index2color = collections.OrderedDict([
        (index, color) for index, color in enumerate(sorted(unique_colors))
    ])
    print("Num classes: ", len(index2color))

    with open(args.out_labeling, "w") as fout:
        json.dump(index2color, fout, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
