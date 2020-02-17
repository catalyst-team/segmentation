#!/usr/bin/env bash
set -e

mkdir -p ./data

download-gdrive 1RCqaQZLziuq1Z4sbMpwD_WHjqR5cdPvh dsb2018_cleared_191109.tar.gz
tar -xf dsb2018_cleared_191109.tar.gz &>/dev/null
mv dsb2018_cleared_191109 ./data/origin

CUDA_VISIBLE_DEVICES="" \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
bash ./bin/catalyst-instance-segmentation-pipeline.sh \
  --config-template ./configs/templates/instance.yml \
  --workdir ./logs \
  --datadir ./data/origin \
  --num-workers 0 \
  --batch-size 2 \
  --max-image-size 256 \
  --check


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

assert aggregated_loss < 0.9
assert iou_soft > 0.05
assert iou_hard > 0.1
"""
