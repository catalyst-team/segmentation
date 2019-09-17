import collections
import json

from catalyst.data.dataset import ListDataset
from catalyst.data.reader import ImageReader, ReaderCompose
from catalyst.dl import ConfigExperiment
from catalyst.utils.pandas import read_csv_data
import torch.nn as nn

from .transforms import Compose, hard_transform, post_transforms, \
    pre_transforms


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage1"]:
            for param in model_.encoder.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in model_.encoder.parameters():
                param.requires_grad = True
        return model_

    @staticmethod
    def get_transforms(
        stage: str = None,
        mode: str = None,
        image_size: int = 256
    ):
        pre_transform_fn = pre_transforms(image_size=image_size)

        if mode == "train":
            post_transform_fn = Compose(
                [hard_transform(image_size=image_size),
                 post_transforms()]
            )
        elif mode in ["valid", "infer"]:
            post_transform_fn = post_transforms()
        else:
            raise NotImplementedError()

        transform_fn = Compose([pre_transform_fn, post_transform_fn])

        def process(dict_):
            result = transform_fn(**dict_)
            return result

        return process

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
        image_size: int = 256
    ):
        datasets = collections.OrderedDict()
        tag2class = json.load(open(tag2class)) \
            if tag2class is not None \
            else None

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
            n_folds=n_folds
        )

        open_fn = ReaderCompose(readers=[
            ImageReader(
                input_key="sample",
                output_key="image",
                datapath=datapath
            ),
            ImageReader(
                input_key="mask",
                output_key="mask",
                datapath=datapath,
                grayscale=True
            ),
        ])

        for mode, source in zip(
            ("train", "valid", "infer"), (df_train, df_valid, df_infer)
        ):
            if len(source) > 0:
                dataset = ListDataset(
                    list_data=source,
                    open_fn=open_fn,
                    dict_transform=self.get_transforms(
                        stage=stage,
                        mode=mode,
                        image_size=image_size
                    )
                )
                datasets[mode] = dataset

        return datasets
