# flake8: noqa
# isort:skip_file

from catalyst.dl import registry, SupervisedRunner as Runner
from catalyst.contrib.models.cv import segmentation as m

from .experiment import Experiment
from . import callbacks

registry.MODELS.add_from_module(m)
registry.CALLBACKS.add_from_module(callbacks)
