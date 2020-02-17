from typing import Set  # isort:skip
import argparse
import collections
import functools
import json
import os
from pathlib import Path

import numpy as np

from catalyst.utils import get_pool, imread, Pool, tqdm_parallel_imap


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


class Preprocessor:
    def __init__(self, in_dir: Path, out_labeling: Path):
        self.in_dir = in_dir
        self.out_labeling = out_labeling

    def preprocess(self, image_path: Path) -> Set:
        image = imread(image_path, rootpath=str(self.in_dir))

        colors = np.unique(image.reshape(-1, image.shape[-1]), axis=0)

        # np.array to hashable
        result = {tuple(row) for row in colors.tolist()}

        return result

    def process_all(self, pool: Pool):
        images = os.listdir(self.in_dir)
        colors = tqdm_parallel_imap(self.preprocess, images, pool)

        unique_colors = functools.reduce(lambda s1, s2: s1 | s2, colors)
        index2color = collections.OrderedDict([
            (index, color) for index, color in enumerate(sorted(unique_colors))
        ])

        print("Num classes: ", len(index2color))

        with open(self.out_labeling, "w") as fout:
            json.dump(index2color, fout, indent=4)


def main(args, _=None):
    args = args.__dict__
    args.pop("command", None)
    num_workers = args.pop("num_workers")

    with get_pool(num_workers) as p:
        Preprocessor(**args).process_all(p)


if __name__ == "__main__":
    args = parse_args()
    main(args)
