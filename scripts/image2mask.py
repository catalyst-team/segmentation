import argparse
import collections
import os
import pandas as pd

from utils import find_images_in_dir, id_from_fname


def build_args(parser):
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="Path to directory with dataset"
    )
    parser.add_argument(
        "--out-dataset",
        type=str,
        required=True,
        help="Path to output dataframe"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    """

    in_dir
    |-- images
    |   |-- sample_1
    |   ..
    |   `-- sample_N
    `-- masks
        |-- sample_1.tiff  # image of shape HxWxM
        ..
        `-- sample_N.tiff  # image of shape HxWxK

    """
    samples = collections.defaultdict(dict)
    for key in ("images", "masks"):
        for fname in find_images_in_dir(os.path.join(args.in_dir, key)):
            fname = os.path.join(key, fname)
            sample_id = id_from_fname(fname)
            samples[sample_id].update({"name": sample_id, key: fname})

    dataframe = pd.DataFrame.from_dict(samples, orient="index")

    isna_row = dataframe.isna().any(axis=1)
    if isna_row.any():
        fname = os.path.join(os.path.basename(args.out_dataset), "nans.json")
        dataframe[isna_row].to_json(fname, orient="records")
        dataframe.dropna(axis=0, how="any", inplace=True)

    print("Num samples: ", dataframe.shape[0])

    dataframe.to_csv(args.out_dataset, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
