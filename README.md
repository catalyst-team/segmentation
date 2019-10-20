[![Build Status](https://travis-ci.com/catalyst-team/segmentation.svg?branch=master)](https://travis-ci.com/catalyst-team/segmentation)
[![Telegram](./pics/telegram.svg)](https://t.me/catalyst_team)
[![Gitter](https://badges.gitter.im/catalyst-team/community.svg)](https://gitter.im/catalyst-team/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Slack](./pics/slack.svg)](https://opendatascience.slack.com/messages/CGK4KQBHD)
[![Donate](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/patreon.png)](https://www.patreon.com/catalyst_team)

# Catalyst.Segmentation

You will learn how to build image segmentation pipeline with transfer learning using the Catalyst framework.

## Goals
1. Install requirements
2. Prepare data
3. Run segmentation pipeline: raw data → production-ready model
4. Get reproducible results

## 1. Install requirements

### Using local environment: 

```bash
pip install -r requirements.txt
```

### Using docker:

This creates a build `catalyst-segmentation` with the necessary libraries:
```bash
make docker-build
```

## 2. Get Dataset

### Try on open datasets

```bash
mkdir data
```
You can use one of the following datasets:

* Binary segmentation: [ISBI](https://biomedicalimaging.org/2015/program/isbi-challenges/)
```bash
    wget https://www.dropbox.com/s/cgf66pl8u0ytow7/isbi_cleared_191013.tar.gz
    tar -xf isbi_cleared_191013.tar.gz &>/dev/null
    mv isbi_cleared_191013 ./data/origin
```

###  Prepare your dataset

#### Data structure
Make sure, that final folder with data has stucture:
```bash
/path/to/your_dataset/
        images/
            image_1
            image_2
            ...
            image_N
        masks/
            mask_1
            mask_2
            ...
            mask_N
```
#### Data location

* The easiest way is to move your data:
    ```bash
    mv /path/to/your_dataset/* /catalyst.segmentation/data/origin 
    ``` 
    In that way you can run pipeline with default settings. 

* If you prefer leave data in `/path/to/your_dataset/` 
    * In local environment:
        * Link directory
            ```bash
            ln -s /path/to/your_dataset $(pwd)/data/origin
            ```
         * Or just set path to your dataset `DATADIR=/path/to/your_dataset` when you start the pipeline.

    * Using docker

        You need to set:
        ```bash
           -v /path/to/your_dataset:/data \ #instead default  $(pwd)/data/origin:/data
         ```
        in the script below to start the pipeline.

## 3. Segmentation pipeline
### Fast&Furious: raw data → production-ready model

The pipeline will automatically guide you from raw data to the production-ready model. 

#### Run in local environment: 

```bash	
CUDA_VISIBLE_DEVICES=0 \	
CUDNN_BENCHMARK="True" \	
CUDNN_DETERMINISTIC="True" \	
WORKDIR=./logs \	
DATADIR=./data/origin \	
IMAGE_SIZE=256 \  # IMAGE_SIZE mod 32 = 0	
CONFIG_TEMPLATE=./configs/templates/binary.yml \	
NUM_WORKERS=4 \	
BATCH_SIZE=256 \	
bash ./bin/catalyst-binary-segmentation-pipeline.sh	
```

#### Run in docker:

```bash
export LOGDIR=$(pwd)/logs
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ -v $LOGDIR:/logdir/  $(pwd)/data/origin:/data \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "USE_WANDB=1" \
   -e "LOGDIR=/logdir" \
   -e "CUDNN_BENCHMARK='True'" \	
   -e "CUDNN_DETERMINISTIC='True'" \	
   -e "WORKDIR=/logdir" \	
   -e "DATADIR=/data" \	
   -e "IMAGE_SIZE=256" \	
   -e "CONFIG_TEMPLATE=./configs/templates/binary.yml" \	
   -e "NUM_WORKERS=4" \	
   -e "BATCH_SIZE=256" \	
   catalyst-segmentation ./bin/catalyst-binary-segmentation-pipeline.sh
```

## 4. Results
All results of all experiments can be found locally in `WORKDIR`, by default `catalyst.segmentation/logs`. Results of experiment, for instance `catalyst.segmentation/logs/logdir-191010-141450-c30c8b84`, contain:

#### checkpoints
*  The directory contains all checkpoints: best, last, also of all stages.
* `best.pth` and `last.pht` can be also found in the corresponding experiment in your W&B account.

#### configs
*  The directory contains experiment\`s configs for reproducibility.

#### logs 
* The directory contains all logs of experiment. 
* Metrics also logs can be displayed in the corresponding experiment in your W&B account.

#### code
*  The directory contains code on which calculations were performed. This is necessary for complete reproducibility.
