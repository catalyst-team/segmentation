#!/usr/bin/env bash
set -e

mkdir -p ./data

download-gdrive 1uyPb9WI0t2qMKIqOjFKMv1EtfQ5FAVEI isbi_cleared_191107.tar.gz
tar -xf isbi_cleared_191107.tar.gz &>/dev/null
mv isbi_cleared_191107 ./data/origin

CUDA_VISIBLE_DEVICES="" \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
./bin/catalyst-binary-segmentation-pipeline.sh \
  --config-template ./configs/templates/binary.yml \
  --workdir ./logs \
  --datadir ./data/origin \
  --num-workers 0 \
  --batch-size 8 \
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

assert aggregated_loss < 1.4
assert iou_soft > 0.25
assert iou_hard > 0.25
"""
