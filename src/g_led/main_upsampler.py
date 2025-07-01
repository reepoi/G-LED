from collections import defaultdict
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import torch
import lightning.pytorch as pl

from conf import conf
from g_led import callbacks, datasets, models, loggers, utils


log = utils.getLoggerByFilename(__file__)


class TrainUpsampler(pl.LightningModule):
    def __init__(self, cfg, downsampler, upsampler, imagen_trainer):
        super().__init__()
        self.automatic_optimization = False
        self.strict_loading = False

        self.cfg = cfg
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.imagen_trainer = imagen_trainer
        self.forecast_time_step_count = 1

    def configure_optimizers(self):
        return None

    def setup(self, stage):
        pass

    def upsample(self, batch_macro):
        if isinstance(self.cfg.model, conf.Trained):
            time_step_window_size = self.cfg.model.conf.model.time_step_window_size
        else:
            time_step_window_size = self.cfg.model.time_step_window_size
        batch_macro_interpolated_to_micro = self.upsampler(batch_macro)
        batch_micro = self.imagen_trainer.sample(
            video_frames=time_step_window_size,
            cond_images=batch_macro_interpolated_to_micro.transpose(1, 2)
        ).transpose(1, 2)
        return batch_micro

    def training_step(self, batch, batch_idx):
        self.optimizers().step()  # increment global step for logging and checkpointing

        batch_macro_interpolated_to_micro = self.upsampler(self.downsampler(batch))
        loss = self.imagen_trainer(
            batch.transpose(1, 2),
            cond_images=batch_macro_interpolated_to_micro.transpose(1, 2),
            unet_number=1,
            ignore_time=False
        )
        self.imagen_trainer.update(unet_number=1)

        return dict(loss=torch.tensor(loss))

    def validation_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        batch, batch_idx, dataset_idx = batch
        batch_macro = self.upsample(self.downsampler(batch))
        return {
            datasets.TrajectoryDataset.dataset_idx_to_dataset_name[dataset_idx]: batch_macro
        }


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

    pl.seed_everything(cfg.rng_seed)
    with pl.utilities.seed.isolate_rng():
        dataset = datasets.get_dataset(cfg.dataset)
        dataset.prepare_data()
    with pl.utilities.seed.isolate_rng():
        imagen_trainer, ckpt_path = models.get_model(cfg)

    downsampler = datasets.Downsampler(cfg.dataset)
    upsampler = datasets.Upsampler(cfg.dataset)
    train_upsampler = TrainUpsampler(cfg, downsampler, upsampler, imagen_trainer)

    logger = loggers.CSVLogger(cfg.run_dir, name=None)

    cbs = [
        callbacks.TimeStepProgressBar(cfg),
        callbacks.LogStats(),
        callbacks.ModelCheckpoint(
            dirpath=cfg.run_dir,
            filename='{epoch}',
            save_last='link',
            monitor='loss',
            save_top_k=2,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        )
    ]
    trainer = pl.Trainer(
        # detect_anomaly=True,
        accelerator=cfg.device,
        devices=1,
        logger=logger,
        max_epochs=cfg.model.epoch_count,
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=cbs,
        # fast_dev_run=2,
        # profiler='simple',
    )

    if cfg.fit:
        trainer.fit(train_upsampler, datamodule=dataset, ckpt_path=ckpt_path)
    if cfg.predict:
        prediction_batches = trainer.predict(train_upsampler, datamodule=dataset, ckpt_path=ckpt_path)
        predictions = defaultdict(list)
        for batch in prediction_batches:
            for k, v in batch.items():
                predictions[k].append(v)
        for k, v in predictions.items():
            torch.save(torch.cat(v).cpu(), cfg.run_dir/f'pred_{k}.pt')


if __name__ == '__main__':
    last_override, run_dir = utils.get_run_dir()
    utils.set_run_dir(last_override, run_dir)
    main()
