#!/usr/bin/env bash
set -e

mkdir -p data

download-gdrive 1MVimVlpc2u2UrZE3Dl-8D1QqG-6HNCKK voc2012_subset_cleared_191109.tar.gz
tar -xf voc2012_subset_cleared_191109.tar.gz &>/dev/null
mv voc2012_subset_cleared_191109 ./data/origin

USE_WANDB=0 \
CUDA_VISIBLE_DEVICES="" \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
WORKDIR=./logs \
DATADIR=./data/origin \
MAX_IMAGE_SIZE=256 \
CONFIG_TEMPLATE=./configs/templates/semantic.yml \
NUM_WORKERS=0 \
BATCH_SIZE=2 \
bash ./bin/catalyst-semantic-segmentation-pipeline.sh --check


python -c """
import pathlib
from safitty import Safict

folder = list(pathlib.Path('./logs/').glob('logdir-*'))[0]
metrics = Safict.load(f'{folder}/checkpoints/_metrics.json')

aggregated_loss = metrics.get('best', 'loss')
iou_soft = metrics.get('best', 'iou_soft')
iou_hard = metrics.get('best', 'iou_hard')

print(aggregated_loss)
print(iou_soft)
print(iou_hard)

assert aggregated_loss < 1.0
assert iou_soft > 0.07
assert iou_hard > 0.03
"""
