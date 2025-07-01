from collections import defaultdict
import logging

import torch
import lightning.pytorch as pl


log = logging.getLogger(__file__)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = '_'

    def check_monitor_top_k(self, trainer: "pl.Trainer", current=None) -> bool:
        if current is None:
            return False

        if self.save_top_k == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        monitor_op = {"min": torch.le, "max": torch.ge}[self.mode]  # changed '<' and '>' to '<=' and '>='
        should_update_best_and_save = monitor_op(current, self.best_k_models[self.kth_best_model_path])

        # If using multiple devices, make sure all processes are unanimous on the decision.
        should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

        return should_update_best_and_save


class TimeStepProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop('v_num', None)
        return items


class LogStatsSequential(pl.callbacks.Callback):
    prefixes = {0: 'val_on_train', 1: 'val'}

    def __init__(self):
        super().__init__()
        self.ckpt_monitor = None
        self.forecast_time_step_count_and_ckpt_path = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        log_kwargs = dict(batch_size=batch.shape[0], on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict(outputs, **log_kwargs)

    def on_validation_start(self, trainer, pl_module):
        self.log('forecast_time_step_count', pl_module.forecast_time_step_count, on_epoch=True, prog_bar=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch, batch_idx, dataset_idx = batch
        log_kwargs = dict(batch_size=batch.shape[0], on_epoch=True, sync_dist=True, prog_bar=True)
        self.log(f'{self.prefixes[dataset_idx]}_relative_rmse_mean', outputs['relative_rmse_mean'], **log_kwargs)
        self.log(f'{self.prefixes[dataset_idx]}_relative_rmse_max', outputs['relative_rmse_max'], reduce_fx='max', **log_kwargs)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        should_increment_forecast_time_step_count = (
            trainer.callback_metrics['val_on_train_relative_rmse_max'] < pl_module.cfg.model.march_tolerance
            or trainer.callback_metrics['val_on_train_relative_rmse_mean'] < (0.1)**(1/2) * pl_module.cfg.model.march_tolerance
        )

        # If using multiple devices, make sure all processes are unanimous on the decision.
        should_increment_forecast_time_step_count = trainer.strategy.reduce_boolean_decision(should_increment_forecast_time_step_count)

        if should_increment_forecast_time_step_count:
            pl_module.lr_schedulers().step()
            pl_module.forecast_time_step_count += 1
            self.ckpt_monitor = None

        current_ckpt_monitor_value = trainer.callback_metrics['val_on_train_relative_rmse_mean']
        should_save_ckpt = should_increment_forecast_time_step_count or (
            self.ckpt_monitor is None or current_ckpt_monitor_value < self.ckpt_monitor
        )
        should_save_ckpt = trainer.strategy.reduce_boolean_decision(should_save_ckpt)
        if should_save_ckpt:
            self.ckpt_monitor = current_ckpt_monitor_value
            current_ckpt_filepath = pl_module.cfg.run_dir/f"epoch_{trainer.current_epoch}__forecast_time_step_count_{trainer.callback_metrics['forecast_time_step_count']:.0f}.ckpt"
            trainer.save_checkpoint(current_ckpt_filepath)
            if self.forecast_time_step_count_and_ckpt_path is not None and self.forecast_time_step_count_and_ckpt_path[0] == trainer.callback_metrics['forecast_time_step_count']:
                trainer.strategy.remove_checkpoint(self.forecast_time_step_count_and_ckpt_path[1])
            self.forecast_time_step_count_and_ckpt_path = (trainer.callback_metrics['forecast_time_step_count'], current_ckpt_filepath)
