import pprint
import sys

from einops import reduce
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from conf import conf
from g_led import callbacks, datasets, loggers, utils
from g_led.transformer.sequentialModel import SequentialModel as Transformer


log = utils.getLoggerByFilename(__file__)


class TrainSequential(pl.LightningModule):
    def __init__(self, cfg, down_sampler, model):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.down_sampler = down_sampler
        self.model = model
        self.forecast_time_step_count = 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.learning_rate)
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.cfg.model.learning_rate_decay)
            ),
        )

    def setup(self, stage):
        pass

    def batch_to_coarse(self, batch):
        batch_size, time_count = batch.shape[:2]
        coarse_batch = self.down_sampler(
            batch.view(-1, self.cfg.dataset.solution_dimension, *self.cfg.dataset.dimensions())
        ).view(batch_size, time_count, self.cfg.dataset.solution_dimension * self.cfg.dataset.embedding_dimension)
        return coarse_batch

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        coarse_batch = self.batch_to_coarse(batch)
        window = coarse_batch[:, :self.cfg.model.time_step_window_size, :]
        window_shifted_by_1_pred, *_ = self.model(inputs_embeds=window, past=None)
        window_shifted_by_1 = coarse_batch[:, 1:self.cfg.model.time_step_window_size+1, :]

        loss = F.mse_loss(window_shifted_by_1_pred, window_shifted_by_1)

        self.manual_backward(loss)
        optimizer.step()

        return dict(loss=loss)

    def validation_step(self, batch, _):
        # if ite_thold is None:
        #     pass
        # else:
        #     if iteration>ite_thold:
        #         break
        # batch = batch.to(args.device).float()
        batch, batch_idx, dataset_idx = batch
        # log.info('Batch %d of validation dataset %d', batch_idx, dataset_idx)
        cached_keys_values = None
        coarse_batch = self.batch_to_coarse(batch)
        window_pred = coarse_batch[:, :1]
        coarse_batch = coarse_batch[:, 1:self.forecast_time_step_count+1]
        window_pred_batch = []
        for j in range(self.forecast_time_step_count):
            if j == 0 or cached_keys_values[0][0].shape[2] < self.cfg.model.time_step_window_size:
                # cached_keys_values[*][0].shape[2] is the number of time steps processed in the trajectory (i.e., in the context of LLMs, the number of tokens in the context)
                window_shifted_by_1_pred, cached_keys_values, *_ = self.model(inputs_embeds=window_pred, past=cached_keys_values)
            else:
                # drop oldest key/value
                cached_keys_values = [
                    [
                        # keys
                        cached_keys_values[layer][0][:, :, 1:],
                        # values
                        cached_keys_values[layer][1][:, :, 1:]
                    ]
                    for layer in range(self.cfg.model.attention_layer_count)
                ]
                window_shifted_by_1_pred, cached_keys_values, *_ = self.model(inputs_embeds=window_pred, past=cached_keys_values)
            window_pred = window_shifted_by_1_pred
            window_pred_batch.append(window_pred)
        window_pred_batch = torch.cat(window_pred_batch, dim=1)

        # local_batch_size = windows_pred.shape[0]
        relative_error_batch = reduce(
            (window_pred_batch - coarse_batch).square(),
            'batch time_step dim -> batch',
            'mean'
        ) / reduce(coarse_batch.square(), 'batch time_step dim -> batch', 'mean')

        if (
            relative_error_batch.max() < self.cfg.model.march_tolerance
            or relative_error_batch.mean() < 0.1 * self.cfg.model.march_tolerance
        ):
            self.forecast_time_step_count += 1
            self.lr_schedulers().step()

        return dict(
            relative_error_max=relative_error_batch.max(),
            relative_error_min=relative_error_batch.min(),
            relative_error_mean=relative_error_batch.mean(),
            relative_error_std=relative_error_batch.std(correction=0),
        )
        # return max(REs),min(REs),sum(REs)/len(REs),3*np.std(np.asarray(REs))


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)

    # time_step_time_logger = loggers.CSVLogger(cfg.run_dir, name=None, name_metrics_file='time_step_times.csv')

    down_sampler = nn.Upsample(size=cfg.dataset.coarse_dimensions(), mode='bilinear')
    model = Transformer(cfg.model, 2 * cfg.dataset.embedding_dimension).to(cfg.device).float()
    train_sequential = TrainSequential(cfg, down_sampler, model)

    logger = loggers.CSVLogger(cfg.run_dir, name=None)

    cbs = [
        callbacks.TimeStepProgressBar(cfg),
        callbacks.LogStats(),
        callbacks.ModelCheckpoint(
            dirpath=cfg.run_dir,
            filename='{epoch}__{forecast_time_step_count:.0f}',
            save_last='link',
            monitor='forecast_time_step_count',
            save_top_k=2,
            save_on_train_epoch_end=False,
            enable_version_counter=False,
        )
    ]
    trainer = pl.Trainer(
        # detect_anomaly=True,
        accelerator=cfg.device,
        devices=1,
        logger=logger,
        max_epochs=-1,
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=cbs,
    )

    trainer.fit(train_sequential, datamodule=datasets.get_dataset(cfg.dataset))


if __name__ == '__main__':
    last_override, run_dir = utils.get_run_dir()
    utils.set_run_dir(last_override, run_dir)
    main()
