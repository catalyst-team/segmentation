#!/usr/bin/env bash
set -e

mkdir -p data

download-gdrive 1uyPb9WI0t2qMKIqOjFKMv1EtfQ5FAVEI isbi_cleared_191107.tar.gz
tar -xf isbi_cleared_191107.tar.gz &>/dev/null
mv isbi_cleared_191107 ./data/origin

USE_WANDB=0 \
CUDA_VISIBLE_DEVICES="" \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
WORKDIR=./logs \
DATADIR=./data/origin \
MAX_IMAGE_SIZE=256 \
CONFIG_TEMPLATE=./configs/templates/binary.yml \
NUM_WORKERS=0 \
BATCH_SIZE=8 \
bash ./bin/catalyst-binary-segmentation-pipeline.sh --check


python -c """
import pathlib
from safitty import Safict

folder = list(pathlib.Path('./logs/').glob('logdir-*'))[0]
metrics = Safict.load(f'{folder}/checkpoints/_metrics.json')

aggregated_loss = metrics.get('best', 'loss')
iou_soft = metrics.get('best', 'iou_soft')
iou_hard = metrics.get('best', 'iou_hard')

assert aggregated_loss < 0.5
assert iou_soft > 0.5
assert iou_hard > 0.65
"""
