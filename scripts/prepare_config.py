# usage:
# python scripts/prepare_config.py \
#     --in-template=./configs/templates/binary.yml \
#     --out-config=./configs/config.yml \
#     --expdir=./src \
#     --dataset-path=./data \
#     --num-workers=4 \
#     --batch-size=64 \
#     --image-size=256

import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def build_args(parser):
    """Constructs the command-line arguments for ``prepare_config``."""
    parser.add_argument("--in-template", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--expdir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)

    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-image-size", default=256, type=int)

    return parser


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def render_config(
    in_template: Path,
    out_config: Path,
    expdir: Path,
    dataset_path: Path,
    num_classes: int,
    num_workers: int,
    batch_size: int,
    max_image_size: int,
):
    """Render catalyst config with specified parameters."""
    template_path = str(in_template.absolute().parent)
    env = Environment(
        loader=FileSystemLoader([template_path]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(in_template.name)

    if num_classes is None:
        with (dataset_path / "index2color.json").open() as f:
            num_classes = len(json.load(f))

    out_config.parent.mkdir(parents=True, exist_ok=True)

    out_config.write_text(
        template.render(
            expdir=str(expdir),
            dataset_path=str(dataset_path),
            num_classes=num_classes,
            num_workers=num_workers,
            batch_size=batch_size,
            max_image_size=max_image_size,
        )
    )


def main(args, _=None):
    """Run the ``prepare_config`` script."""
    args = args.__dict__
    render_config(**args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
