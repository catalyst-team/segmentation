import argparse
import collections
import os
from pathlib import Path

import pandas as pd

from catalyst.contrib.utils import has_image_extension


def build_args(parser):
    """Constructs the command-line arguments for ``image2mask``."""
    parser.add_argument(
        "--in-dir",
        required=True,
        type=Path,
        help="Path to directory with dataset",
    )
    parser.add_argument(
        "--out-dataset",
        required=True,
        type=Path,
        help="Path to output dataframe",
    )

    return parser


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    """Run the ``image2mask`` script. # noqa: RST202, RST214, RST215, RST299

    Args:
        args: CLI args:
            * `in-dir`: Path to dataset, `in-dir` folder structure:
            ```
                in_dir
                |-- images
                |   |-- sample_1
                |   ..
                |   `-- sample_N
                `-- masks
                    |-- sample_1.tiff  # image of shape HxWxM
                    ..
                    `-- sample_N.tiff  # image of shape HxWxK
            ```
            * `out-dataset`: Path to output dataframe.
    """
    samples = collections.defaultdict(dict)
    for key in ("images", "masks"):
        for filename in (args.in_dir / key).iterdir():
            if has_image_extension(str(filename)):
                sample_id = filename.stem
                samples[sample_id].update(
                    {"name": sample_id, key: str(filename)}
                )

    dataframe = pd.DataFrame.from_dict(samples, orient="index")

    isna_row = dataframe.isna().any(axis=1)
    if isna_row.any():
        fname = os.path.join(args.out_dataset.parent, "nans.json")
        dataframe[isna_row].to_json(fname, orient="records")
        dataframe.dropna(axis=0, how="any", inplace=True)

    print("Num samples: ", dataframe.shape[0])

    dataframe.to_csv(args.out_dataset, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
