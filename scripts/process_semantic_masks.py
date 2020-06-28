import argparse
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import safitty

from catalyst.contrib.utils import (
    get_pool,
    has_image_extension,
    imread,
    mimwrite_with_meta,
    tqdm_parallel_imap,
)


def build_args(parser):
    """Constructs the command-line arguments for ``process_semantic_masks``."""
    parser.add_argument(
        "--in-dir", required=True, type=Path, help="Raw masks folder path"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Processed masks folder path",
    )
    parser.add_argument("--index2color", required=True, type=Path)
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Number of workers to parallel the processing",
    )

    return parser


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


class Preprocessor:
    """Multi-thread mask pre-processor."""

    def __init__(self, in_dir: Path, out_dir: Path, index2color: Path):
        """Constructor method for the :class:`Preprocessor` class.

        Args:
            in_dir (Path): path to folder with input masks
            out_dir (Path): path to folder to store processed masks
            index2color (Path): path to file with mapping from mask index
                to mask color, look at COCO dataset for details
        """
        self.in_dir = in_dir
        self.out_dir = out_dir

        index2color = safitty.load(args.index2color)
        self.index2color = {
            int(key): tuple(value) for key, value in index2color.items()
        }

    def preprocess(self, image_path: Path) -> None:
        """Process single mask.

        Args:
            image_path (Path): path to the mask
        """
        image = imread(image_path, rootpath=str(self.in_dir))
        heigth, width = image.shape[:2]

        mask = np.zeros((heigth, width, len(self.index2color)), dtype=np.uint8)
        for index, color in self.index2color.items():
            mask[np.all((image == color), axis=-1), index] = 255

        target_path = self.out_dir / f"{image_path.stem}.tiff"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        mimwrite_with_meta(
            target_path, np.dsplit(mask, mask.shape[2]), {"compress": 9}
        )

    def process_all(self, pool: Pool):
        """Process all masks."""
        images = [
            filename
            for filename in self.in_dir.iterdir()
            if has_image_extension(str(filename))
        ]
        tqdm_parallel_imap(self.preprocess, images, pool)


def main(args, _=None):
    """Run the ``process_semantic_masks`` script."""
    args = args.__dict__
    args.pop("command", None)
    num_workers = args.pop("num_workers")

    with get_pool(num_workers) as p:
        Preprocessor(**args).process_all(p)


if __name__ == "__main__":
    args = parse_args()
    main(args)
