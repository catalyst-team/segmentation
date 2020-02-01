#!/usr/bin/env bash
#title           :catalyst-instance-segmentation-pipeline
#description     :catalyst.dl script for instance segmentation pipeline run
#author          :Sergey Kolesnikov, Yauheni Kachan
#author_email    :scitator@gmail.com, yauheni.kachan@gmail.com
#date            :20191109
#version         :19.11.1
#==============================================================================

# usage:
# WORKDIR=/path/to/logdir \
# DATADIR=/path/to/dataset \
# IMAGE_SIZE=... \
# CONFIG_TEMPLATE=... \ # model config to use
# ./bin/catalyst-instance-segmentation-pipeline.sh

# example:
# CUDA_VISIBLE_DEVICES=0 \
# CUDNN_BENCHMARK="True" \
# CUDNN_DETERMINISTIC="True" \
# WORKDIR=./logs \
# DATADIR=./data/origin \
# IMAGE_SIZE=256 \
# CONFIG_TEMPLATE=./configs/templates/instance.yml \
# NUM_WORKERS=4 \
# BATCH_SIZE=256 \
# ./bin/catalyst-instance-segmentation-pipeline.sh

set -e

# --- test part
# uncomment and run bash ./bin/catalyst-instance-segmentation-pipeline.sh

#mkdir -p ./data
#download-gdrive 1RCqaQZLziuq1Z4sbMpwD_WHjqR5cdPvh dsb2018_cleared_191109.tar.gz
#tar -xf dsb2018_cleared_191109.tar.gz &>/dev/null
#mv dsb2018_cleared_191109 ./data/origin
#
#export CUDNN_BENCHMARK="True"
#export CUDNN_DETERMINISTIC="True"
#
#export CONFIG_TEMPLATE=./configs/templates/instance.yml
#export WORKDIR=./logs
#export DATADIR=./data/origin
#export NUM_WORKERS=4
#export BATCH_SIZE=64
#export IMAGE_SIZE=256

# ---- environment variables

if [[ -z "$NUM_WORKERS" ]]; then
      NUM_WORKERS=4
fi

if [[ -z "$BATCH_SIZE" ]]; then
      BATCH_SIZE=64
fi

if [[ -z "$IMAGE_SIZE" ]]; then
      IMAGE_SIZE=256
fi

if [[ -z "$CONFIG_TEMPLATE" ]]; then
      CONFIG_TEMPLATE="./configs/templates/instance.yml"
fi

if [[ -z "$DATADIR" ]]; then
      DATADIR="./data/origin"
fi

if [[ -z "$WORKDIR" ]]; then
      WORKDIR="./logs"
fi

SKIPDATA=""
while getopts ":s" flag; do
  case "${flag}" in
    s) SKIPDATA="true" ;;
  esac
done

date=$(date +%y%m%d-%H%M%S)
postfix=$(openssl rand -hex 4)
logname="$date-$postfix"
export DATASET_DIR=$WORKDIR/dataset
export RAW_MASKS_DIR=$DATASET_DIR/raw_masks
export CONFIG_DIR=$WORKDIR/configs-${logname}
export LOGDIR=$WORKDIR/logdir-${logname}

mkdir -p $WORKDIR
mkdir -p $DATASET_DIR
mkdir -p $CONFIG_DIR
mkdir -p $LOGDIR

# ---- data preparation

if [[ -z "${SKIPDATA}" ]]; then
    cp -R $DATADIR/* $DATASET_DIR/

    mkdir -p $DATASET_DIR/masks
    python scripts/process_instance_masks.py \
        --in-dir $RAW_MASKS_DIR \
        --out-dir $DATASET_DIR/masks \
        --num-workers $NUM_WORKERS

    python scripts/image2mask.py \
        --in-dir $DATASET_DIR \
        --out-dataset $DATASET_DIR/dataset_raw.csv

    catalyst-data split-dataframe \
        --in-csv $DATASET_DIR/dataset_raw.csv \
        --n-folds=5 --train-folds=0,1,2,3 \
        --out-csv=$DATASET_DIR/dataset.csv
fi


# ---- config preparation

python ./scripts/prepare_config.py \
    --in-template=$CONFIG_TEMPLATE \
    --out-config=$CONFIG_DIR/config.yml \
    --expdir=./src \
    --dataset-path=$DATASET_DIR \
    --num-classes=2 \
    --num-workers=$NUM_WORKERS \
    --batch-size=$BATCH_SIZE \
    --image-size=$IMAGE_SIZE

cp -r ./configs/_common.yml $CONFIG_DIR/_common.yml


# ---- model training

catalyst-dl run \
    -C $CONFIG_DIR/_common.yml $CONFIG_DIR/config.yml \
    --logdir $LOGDIR $*
