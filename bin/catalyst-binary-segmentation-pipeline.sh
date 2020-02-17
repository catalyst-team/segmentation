#!/usr/bin/env bash
#title           :catalyst-binary-segmentation-pipeline
#description     :catalyst.dl script for full binary segmentation pipeline run
#author          :Sergey Kolesnikov, Yauheni Kachan
#author_email    :scitator@gmail.com, yauheni.kachan@gmail.com
#date            :20191016
#version         :19.10.2
#==============================================================================

set -e

usage()
{
  cat << USAGE >&2
Usage: $(basename "$0") [OPTION...] [catalyst-dl run args...]

  -s, --skipdata                       Skip data preparation
  -j, --num-workers NUM_WORKERS        Number of data loading/processing workers
  -b, --batch-size BATCH_SIZE          Mini-batch size
  --max-image-size MAX_IMAGE_SIZE      Target size of images e.g. 256
  --config-template CONFIG_TEMPLATE    Model config to use
  --datadir DATADIR
  --workdir WORKDIR
  catalyst-dl run args                 Execute \`catalyst-dl run\` with args

Example:
  CUDA_VISIBLE_DEVICES=0 \\
  CUDNN_BENCHMARK="True" \\
  CUDNN_DETERMINISTIC="True" \\
  ./bin/catalyst-binary-segmentation-pipeline.sh \\
    --workdir ./logs \\
    --datadir ./data/origin \\
    --max-image-size 256 \\
    --config-template ./configs/templates/binary.yml \\
    --num-workers 4 \\
    --batch-size 256
USAGE
  exit 1
}


# ---- environment variables

NUM_WORKERS=${NUM_WORKERS:=4}
BATCH_SIZE=${BATCH_SIZE:=64}
MAX_IMAGE_SIZE=${MAX_IMAGE_SIZE:=256}
CONFIG_TEMPLATE=${CONFIG_TEMPLATE:="./configs/templates/binary.yml"}
DATADIR=${DATADIR:="./data/origin"}
WORKDIR=${WORKDIR:="./logs"}
SKIPDATA=""
_run_args=""
while (( "$#" )); do
  case "$1" in
    -j|--num-workers)
      NUM_WORKERS=$2
      shift 2
      ;;
    -b|--batch-size)
      BATCH_SIZE=$2
      shift 2
      ;;
    --max-image-size)
      MAX_IMAGE_SIZE=$2
      shift 2
      ;;
    --config-template)
      CONFIG_TEMPLATE=$2
      shift 2
      ;;
    --datadir)
      DATADIR=$2
      shift 2
      ;;
    --workdir)
      WORKDIR=$2
      shift 2
      ;;
    -s|--skipdata)
      SKIPDATA="true"
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      _run_args="${_run_args} $1"
      shift
      ;;
  esac
done

date=$(date +%y%m%d-%H%M%S)
postfix=$(openssl rand -hex 4)
logname="${date}-${postfix}"
export DATASET_DIR=${WORKDIR}/dataset
export CONFIG_DIR=${WORKDIR}/configs-${logname}
export LOGDIR=${WORKDIR}/logdir-${logname}

for dir in ${WORKDIR} ${DATASET_DIR} ${CONFIG_DIR} ${LOGDIR}; do
  mkdir -p ${dir}
done


# ---- data preparation

if [[ -z "${SKIPDATA}" ]]; then
  cp -R ${DATADIR}/* ${DATASET_DIR}/

  mv ${DATASET_DIR}/raw_masks ${DATASET_DIR}/masks

  python scripts/image2mask.py \
    --in-dir ${DATASET_DIR} \
    --out-dataset ${DATASET_DIR}/dataset_raw.csv

  catalyst-data split-dataframe \
    --in-csv ${DATASET_DIR}/dataset_raw.csv \
    --n-folds=5 --train-folds=0,1,2,3 \
    --out-csv=${DATASET_DIR}/dataset.csv
fi


# ---- config preparation

python ./scripts/prepare_config.py \
  --in-template=${CONFIG_TEMPLATE} \
  --out-config=${CONFIG_DIR}/config.yml \
  --expdir=./src \
  --dataset-path=${DATASET_DIR} \
  --num-classes=1 \
  --num-workers=${NUM_WORKERS} \
  --batch-size=${BATCH_SIZE} \
  --max-image-size=${MAX_IMAGE_SIZE}

cp -r ./configs/_common.yml ${CONFIG_DIR}/_common.yml


# ---- model training

catalyst-dl run \
  -C ${CONFIG_DIR}/_common.yml ${CONFIG_DIR}/config.yml \
  --logdir ${LOGDIR} ${_run_args}
