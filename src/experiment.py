from typing import Optional
import collections
import json

import torch
from torch import nn

from catalyst.contrib.data.cv import ImageReader, MaskReader
from catalyst.contrib.utils.pandas import read_csv_data
from catalyst.data import ListDataset, ReaderCompose, ScalarReader
from catalyst.dl import ConfigExperiment


class Experiment(ConfigExperiment):
    """Segmentation experiment."""

    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )

        if stage in ["debug", "stage1"]:
            for param in model.encoder.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in model.encoder.parameters():
                param.requires_grad = True
        return model

    def get_datasets(
        self,
        stage: str,
        datapath: Optional[str] = None,
        in_csv: Optional[str] = None,
        in_csv_train: Optional[str] = None,
        in_csv_valid: Optional[str] = None,
        in_csv_infer: Optional[str] = None,
        train_folds: Optional[str] = None,
        valid_folds: Optional[str] = None,
        tag2class: Optional[str] = None,
        class_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        folds_seed: int = 42,
        n_folds: int = 5,
    ):
        """Returns the datasets for a given stage and epoch.

        Args:
            stage (str): stage name of interest,
                like "pretrain" / "train" / "finetune" / etc
            datapath (str): path to folder with images and masks
            in_csv (Optional[str]): path to CSV annotation file. Look at
                :func:`catalyst.contrib.utils.pandas.read_csv_data` for details
            in_csv_train (Optional[str]): path to CSV annotaion file
                with train samples.
            in_csv_valid (Optional[str]): path to CSV annotaion file
                with the validation samples
            in_csv_infer (Optional[str]): path to CSV annotaion file
                with test samples
            train_folds (Optional[str]): folds to use for training
            valid_folds (Optional[str]): folds to use for validation
            tag2class (Optional[str]): path to JSON file with mapping from
                class name (tag) to index
            class_column (Optional[str]): name of class index column in the CSV
            tag_column (Optional[str]): name of class name in the CSV file
            folds_seed (int): random seed to use
            n_folds (int): number of folds on which data will be split

        Returns:
            Dict: dictionary with datasets for current stage.
        """
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
