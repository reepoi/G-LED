from collections import defaultdict
import logging

import lightning.pytorch as pl


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
    prefixes = {0: 'val_on_train', 1: 'val'}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_dict(outputs, on_epoch=True, prog_bar=True, batch_size=batch.shape[0])

    def on_validation_start(self, trainer, pl_module):
        self.log('forecast_time_step_count', pl_module.forecast_time_step_count, on_epoch=True, prog_bar=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch, batch_idx, dataset_idx = batch
        outputs = {f'{self.prefixes[dataset_idx]}_{k}': v for k, v in outputs.items() if k != 'relative_rmse_sum'}
        self.log_dict(outputs, on_epoch=True, prog_bar=True, batch_size=batch.shape[0])


class MetricMonitorLRSchedulerStepper(pl.callbacks.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._metrics = defaultdict(lambda: 0)
        self.prefixes = {0: 'val_on_train', 1: 'val'}

    def on_validation_start(self, trainer, pl_module):
        self._metrics = defaultdict(lambda: 0)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch, _, dataset_idx = batch
        self._metrics['data_count'] += batch.shape[0]
        if self.prefixes[dataset_idx] == 'val_on_train':
            self._metrics['max'] = max(self._metrics['max'], outputs['relative_rmse_max'])
            self._metrics['sum'] += outputs['relative_rmse_sum']

    def on_validation_end(self, trainer, pl_module):
        self._metrics['mean'] = self._metrics['sum'] / self._metrics['data_count']
        if (
            self._metrics['max'] < self.cfg.model.march_tolerance
            or self._metrics['mean'] < (0.1)**(1/2) * self.cfg.model.march_tolerance
        ):
            pl_module.lr_schedulers().step()
            pl_module.forecast_time_step_count += 1
