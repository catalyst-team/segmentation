import argparse
import os
import numpy as np

import safitty
from tqdm import tqdm

from catalyst.utils import boolean_flag, imread, mimwrite_with_meta

from utils import id_from_fname


def build_args(parser):
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--index2color", type=str, required=True)
    boolean_flag(parser, "verbose", default=False)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    index2color = safitty.load(args.index2color)
    index2color = {int(key): value for key, value in index2color.items()}
    n_colors = len(index2color)

    os.makedirs(args.out_dir, exist_ok=True)
    for fname in tqdm(os.listdir(args.in_dir), disable=(not args.verbose)):
        image = imread(fname, rootpath=args.in_dir)
        heigth, width = image.shape[:2]
        mask = np.zeros((heigth, width, n_colors), dtype=np.uint8)
        for index in range(n_colors):
            mask[np.all((image == index2color[index]), axis=-1), index] = 255

        mimwrite_with_meta(
            f"{args.out_dir}/{id_from_fname(fname)}.tiff",
            np.dsplit(mask, mask.shape[2]),
            {"compress": 9}
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
