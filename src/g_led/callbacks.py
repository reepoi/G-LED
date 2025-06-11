import logging
import math

import torch
from einops import rearrange, reduce
import lightning.pytorch as pl
import pandas as pd
import polars

from g_led import utils


log = logging.getLogger(__file__)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = '_'


class TimeStepProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop('v_num', None)
        return items


class LogStats(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_dict(outputs, on_epoch=True, prog_bar=True, batch_size=batch.shape[0])

    def on_validation_start(self, trainer, pl_module):
        self.log('forecast_time_step_count', pl_module.forecast_time_step_count, on_epoch=True, prog_bar=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch, batch_idx, dataset_idx = batch
        prefixes = {0: 'val_on_train', 1: 'val'}
        outputs = {f'{prefixes[dataset_idx]}_{k}': v for k, v in outputs.items()}
        self.log_dict(outputs, on_epoch=True, prog_bar=True, batch_size=batch.shape[0])
