#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  DATA  ####################################
rm -rf ./data

# load the data
mkdir -p ./data

# load VOC2012 dataset subset
download-gdrive 1MVimVlpc2u2UrZE3Dl-8D1QqG-6HNCKK voc2012_subset_cleared_191109.tar.gz
tar -xf voc2012_subset_cleared_191109.tar.gz &>/dev/null
mv voc2012_subset_cleared_191109 ./data/origin

# load full VOC2012 dataset
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# tar -xf VOCtrainval_11-May-2012.tar &>/dev/null
# mv VOCdevkit/VOC2012 ./data/origin && \
# mv ./data/origin/JPEGImages ./data/origin/images && \
# mv ./data/origin/SegmentationClass ./data/origin/raw_masks


################################  pipeline 00  ################################
rm -rf ./logs


################################  pipeline 01  ################################
CUDA_VISIBLE_DEVICES="" \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
bash ./bin/catalyst-semantic-segmentation-pipeline.sh \
  --config-template ./configs/templates/semantic.yml \
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

assert aggregated_loss < 1.0
assert iou_soft > 0.07
assert iou_hard > 0.03
"""


################################  pipeline 99  ################################
rm -rf ./logs
