import collections
import json

import torch
import torch.nn as nn

from catalyst.contrib.utils.pandas import read_csv_data
from catalyst.data import (
    ImageReader, ListDataset, MaskReader, ReaderCompose, ScalarReader
)
from catalyst.dl import ConfigExperiment


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage1"]:
            for param in model_.encoder.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in model_.encoder.parameters():
                param.requires_grad = True
        return model_

    def get_datasets(
        self,
        stage: str,
        datapath: str = None,
        in_csv: str = None,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        in_csv_infer: str = None,
        train_folds: str = None,
        valid_folds: str = None,
        tag2class: str = None,
        class_column: str = None,
        tag_column: str = None,
        folds_seed: int = 42,
        n_folds: int = 5,
    ):
        datasets = collections.OrderedDict()
        tag2class = (
            json.load(open(tag2class)) if tag2class is not None else None
        )

        df, df_train, df_valid, df_infer = read_csv_data(
            in_csv=in_csv,
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            train_folds=train_folds,
            valid_folds=valid_folds,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column,
            seed=folds_seed,
            n_folds=n_folds,
        )

        open_fn = ReaderCompose(
            readers=[
                ImageReader(
                    input_key="images", output_key="image", rootpath=datapath
                ),
                MaskReader(
                    input_key="masks", output_key="mask", rootpath=datapath
                ),
                ScalarReader(
                    input_key="name",
                    output_key="name",
                    dtype=str,
                    default_value=-1,
                ),
            ]
        )

        for mode, source in zip(
            ("train", "valid", "infer"), (df_train, df_valid, df_infer)
        ):
            if source is not None and len(source) > 0:
                datasets[mode] = ListDataset(
                    list_data=source,
                    open_fn=open_fn,
                    dict_transform=self.get_transforms(
                        stage=stage, dataset=mode
                    ),
                )

        return datasets
