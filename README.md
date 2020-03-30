<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

**Accelerated DL & RL**

[![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Segmentation_Tests/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Segmentation&tab=projectOverview&guest=1)
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)
[![Github contributors](https://img.shields.io/github/contributors/catalyst-team/catalyst.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/catalyst/graphs/contributors)

[![Twitter](https://img.shields.io/badge/news-on%20twitter-499feb)](https://twitter.com/catalyst_core)
[![Telegram](https://img.shields.io/badge/channel-on%20telegram-blue)](https://t.me/catalyst_team)
[![Spectrum](https://img.shields.io/badge/chat-on%20spectrum-blueviolet)](https://spectrum.chat/catalyst)
[![Slack](https://img.shields.io/badge/ODS-slack-red)](https://opendatascience.slack.com/messages/CGK4KQBHD)


</div>

PyTorch framework for DL & RL research.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather than write another regular train loop. <br/>
Break the cycle - use the Catalyst!

Part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/). Part of [Catalyst Ecosystem](https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing). Project [manifest](https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md).

---

# Catalyst.Segmentation [![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Segmentation_Tests/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Segmentation&tab=projectOverview&guest=1) [![Github contributors](https://img.shields.io/github/contributors/catalyst-team/segmentation.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/segmentation/graphs/contributors)

You will learn how to build image segmentation pipeline with transfer learning using the Catalyst framework.

## Goals
1. Install requirements
2. Prepare data
3. **Run: raw data → production-ready model**
4. **Get results**
5. Customize own pipeline

## 1. Install requirements

### Using local environment:

```bash
pip install -r requirements/requirements.txt
```

### Using docker:

This creates a build `catalyst-segmentation` with the necessary libraries:
```bash
make docker-build
```

## 2. Get Dataset

### Try on open datasets

<details>
<summary>You can use one of the open datasets </summary>
<p>

```bash
export DATASET="isbi"

rm -rf data/
mkdir -p data

if [[ "$DATASET" == "isbi" ]]; then
    # binary segmentation
    # http://brainiac2.mit.edu/isbi_challenge/
    download-gdrive 1uyPb9WI0t2qMKIqOjFKMv1EtfQ5FAVEI isbi_cleared_191107.tar.gz
    tar -xf isbi_cleared_191107.tar.gz &>/dev/null
    mv isbi_cleared_191107 ./data/origin
elif [[ "$DATASET" == "voc2012" ]]; then
    # semantic segmentation
    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar -xf VOCtrainval_11-May-2012.tar &>/dev/null
    mkdir -p ./data/origin/images/; mv VOCdevkit/VOC2012/JPEGImages/* $_
    mkdir -p ./data/origin/raw_masks; mv VOCdevkit/VOC2012/SegmentationClass/* $_
elif [[ "$DATASET" == "dsb2018" ]]; then
    # instance segmentation
    # https://www.kaggle.com/c/data-science-bowl-2018
    download-gdrive 1RCqaQZLziuq1Z4sbMpwD_WHjqR5cdPvh dsb2018_cleared_191109.tar.gz
    tar -xf dsb2018_cleared_191109.tar.gz &>/dev/null
    mv dsb2018_cleared_191109 ./data/origin
fi
```

</p>
</details>

### Use your own dataset

<details>
<summary>Prepare your dataset</summary>
<p>

#### Data structure

Make sure, that final folder with data has the required structure:

<details open>
<summary>Data structure for binary segmentation</summary>
<p>

```bash
/path/to/your_dataset/
        images/
            image_1
            image_2
            ...
            image_N
        raw_masks/
            mask_1
            mask_2
            ...
            mask_N
```
where each `mask` is a binary image

</p>
</details>

<details>
<summary>Data structure for semantic segmentation</summary>
<p>

```bash
/path/to/your_dataset/
        images/
            image_1
            image_2
            ...
            image_N
        raw_masks/
            mask_1
            mask_2
            ...
            mask_N
```
where each `mask` is an image with class encoded through colors e.g. [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset where `bicycle` class is encoded with <span style="color:rgb(0, 128, 0)">green</span> color and `bird` with <span style="color:rgb(128, 128, 0)">olive</span>

</p>
</details>

<details>
<summary>Data structure for instance segmentation</summary>
<p>

```bash
/path/to/your_dataset/
        images/
            image_1
            image_2
            ...
            image_M
        raw_masks/
            mask_1/
                instance_1
                instance_2
                ...
                instance_N
            mask_2/
                instance_1
                instance_2
                ...
                instance_K
            ...
            mask_M/
                instance_1
                instance_2
                ...
                instance_Z
```
where each `mask` represented as a folder with instances images (one image per instance), and masks may consisting of a different number of instances e.g. [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018) dataset

</p>
</details>

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

</p>
</details>

## 3. Segmentation pipeline

### Fast&Furious: raw data → production-ready model

The pipeline will automatically guide you from raw data to the production-ready model.

We will initialize [Unet](https://arxiv.org/abs/1505.04597) model with a pre-trained [ResNet-18](https://arxiv.org/abs/1512.03385) encoder. During current pipeline model will be trained sequentially in two stages.

<details open>
<summary>Binary segmentation pipeline</summary>
<p>

#### Run in local environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
bash ./bin/catalyst-binary-segmentation-pipeline.sh \
    --workdir ./logs \
    --datadir ./data/origin \
    --max-image-size 256 \
    --config-template ./configs/templates/binary.yml \
    --num-workers 4 \
    --batch-size 256
```

#### Run in docker:

```bash
docker run -it --rm --shm-size 8G --runtime=nvidia \
    -v $(pwd):/workspace/ \
    -v $(pwd)/logs:/logdir/ \
    -v $(pwd)/data/origin:/data \
    -e "CUDA_VISIBLE_DEVICES=0" \
    -e "CUDNN_BENCHMARK='True'" \
    -e "CUDNN_DETERMINISTIC='True'" \
    catalyst-segmentation ./bin/catalyst-binary-segmentation-pipeline.sh \
        --workdir /logdir \
        --datadir /data \
        --max-image-size 256 \
        --config-template ./configs/templates/binary.yml \
        --num-workers 4 \
        --batch-size 256
```

</p>
</details>

<details>
<summary>Semantic segmentation pipeline</summary>
<p>

#### Run in local environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
bash ./bin/catalyst-semantic-segmentation-pipeline.sh \
    --workdir ./logs \
    --datadir ./data/origin \
    --max-image-size 256 \
    --config-template ./configs/templates/semantic.yml \
    --num-workers 4 \
    --batch-size 256
```

#### Run in docker:

```bash
docker run -it --rm --shm-size 8G --runtime=nvidia \
    -v $(pwd):/workspace/ \
    -v $(pwd)/logs:/logdir/ \
    -v $(pwd)/data/origin:/data \
    -e "CUDA_VISIBLE_DEVICES=0" \
    -e "CUDNN_BENCHMARK='True'" \
    -e "CUDNN_DETERMINISTIC='True'" \
    catalyst-segmentation ./bin/catalyst-semantic-segmentation-pipeline.sh \
        --workdir /logdir \
        --datadir /data \
        --max-image-size 256 \
        --config-template ./configs/templates/semantic.yml \
        --num-workers 4 \
        --batch-size 256
```

</p>
</details>

<details>
<summary>Instance segmentation pipeline</summary>
<p>

#### Run in local environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
bash ./bin/catalyst-semantic-segmentation-pipeline.sh \
    --workdir ./logs \
    --datadir ./data/origin \
    --max-image-size 256 \
    --config-template ./configs/templates/instance.yml \
    --num-workers 4 \
    --batch-size 256
```

#### Run in docker:

```bash
docker run -it --rm --shm-size 8G --runtime=nvidia \
    -v $(pwd):/workspace/ \
    -v $(pwd)/logs:/logdir/ \
    -v $(pwd)/data/origin:/data \
    -e "CUDA_VISIBLE_DEVICES=0" \
    -e "CUDNN_BENCHMARK='True'" \
    -e "CUDNN_DETERMINISTIC='True'" \
    catalyst-segmentation ./bin/catalyst-instance-segmentation-pipeline.sh \
        --workdir /logdir \
        --datadir /data \
        --max-image-size 256 \
        --config-template ./configs/templates/instance.yml \
        --num-workers 4 \
        --batch-size 256
```

</p>
</details>

The pipeline is running and you don’t have to do anything else, it remains to wait for the best model!

#### Visualizations

Tensorboard can be used for visualisation:

```bash
tensorboard --logdir=/catalyst.segmentation/logs
```
<img src="/pics/tf_metrics.png" title="tf binary segmentation metrics"  align="left">

## 4. Results
All results of all experiments can be found locally in `WORKDIR`, by default `catalyst.segmentation/logs`. Results of experiment, for instance `catalyst.segmentation/logs/logdir-191107-094627-2f31d790`, contain:

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

## 5. Customize own pipeline

For your future experiments framework provides powerful configs allow to optimize configuration of the whole pipeline of segmentation in a controlled and reproducible way.

<details>
<summary>Configure your experiments</summary>
<p>

* Common settings of stages of training and model parameters can be found in `catalyst.segmentation/configs/_common.yml`.
    * `model_params`: detailed configuration of models, including:
        * model, for instance `ResNetUnet`
        * detailed architecture description
        * using pretrained model
    * `stages`: you can configure training or inference in several stages with different hyperparameters. In our example:
        * optimizer params
        * first learn the head(s), then train the whole network

* The `CONFIG_TEMPLATE` with other experiment\`s hyperparameters, such as data_params and is here: `catalyst.segmentation/configs/templates/binary.yml`.  The config allows you to define:
    * `data_params`: path, batch size, num of workers and so on
    * `callbacks_params`: Callbacks are used to execute code during training, for example, to get metrics or save checkpoints. Catalyst provide wide variety of helpful callbacks also you can use custom.

You can find much more options for configuring experiments in [catalyst documentation.](https://catalyst-team.github.io/catalyst/)

</p>
</details>
