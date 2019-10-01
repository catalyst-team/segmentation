import argparse
import collections
import logging
import os
import pandas as pd

from pytorch_toolbelt.utils.fs import find_images_in_dir, id_from_fname

logger = logging.getLogger(__name__)


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
    samples = collections.defaultdict(dict)
    for key in ("images", "masks"):
        for fname in find_images_in_dir(os.path.join(args.in_dir, key)):
            sample_id = id_from_fname(fname)
            samples[sample_id].update({"name": sample_id, key: fname})

    dataframe = pd.DataFrame.from_dict(samples, orient="index")

    # check for missed values
    if dataframe.isna().any(axis=None):
        missed_values = dataframe.isna().any(axis=1)
        for row in dataframe[missed_values].to_dict("records"):
            logger.warning(
                f"sample `{row['id']}`: some of the values are missed ({row})"
            )

        # drop rows with missing values
        dataframe = dataframe.drop(missed_values[missed_values].index)

    print("Num samples: ", dataframe.shape[0])

    dataframe.to_csv(args.out_dataset, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
