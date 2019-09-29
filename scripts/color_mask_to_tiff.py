import argparse
import os
import numpy as np
import imageio

from pytorch_toolbelt.utils.fs import id_from_fname
import safitty
from tqdm import tqdm

from catalyst.utils import mimwrite_with_meta


def build_args(parser):
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print additional information"
    )
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    config = safitty.load(args.config)
    annotation = config["mapping"]
    os.makedirs(args.out_dir, exist_ok=True)

    for fname in tqdm(os.listdir(args.in_dir), disable=(not args.verbose)):
        image = imageio.imread(
            os.path.join(args.in_dir, fname), pilmode=config["pilmode"]
        )
        heigth, width = image.shape[:2]
        mask = np.zeros((heigth, width, len(annotation)), dtype=np.uint8)
        for index, (class_name, class_params) in enumerate(annotation.items()):
            label = class_params.get("label", index)
            color = class_params.get("color", label)
            mask[np.all((image == color), axis=-1), label] = 255

        mimwrite_with_meta(
            f"{args.out_dir}/{id_from_fname(fname)}.tiff",
            np.dsplit(mask, mask.shape[2]),
            {"compress": 9}
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
