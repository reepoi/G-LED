import pprint
import sys

from einops import reduce
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from imagen_pytorch import ElucidatedImagen, ImagenTrainer, Unet3D, Unet

from conf import conf
from g_led import callbacks, datasets, loggers, utils


log = utils.getLoggerByFilename(__file__)


class TrainUpsampler(pl.LightningModule):
    def __init__(self, cfg, downsampler, upsampler, imagen_trainer):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.imagen_trainer = imagen_trainer
        self.forecast_time_step_count = 1

    def configure_optimizers(self):
        return None

    def setup(self, stage):
        pass

    def training_step(self, batch, batch_idx):
        batch_macro = self.downsampler(
            batch.view(-1, self.cfg.dataset.solution_dimension, *self.cfg.dataset.dimensions())
        )
        batch_macro_interpolated_to_micro = self.upsampler(batch_macro).view(batch.shape)

        batch = batch.transpose(1, 2)[..., None]
        batch_macro_interpolated_to_micro = batch_macro_interpolated_to_micro.transpose(1, 2)[..., None]

        loss = self.imagen_trainer(
            batch,
            cond_images=batch_macro_interpolated_to_micro,
            unet_number=1,
            ignore_time=False
        )
        self.imagen_trainer.update(unet_number=1)

        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        pass


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
        unet1 = Unet3D(
            dim=32,  # diff_args.unet_dim,
            cond_images_channels=1,
            memory_efficient=True,
            dim_mults=(1, 2, 4, 8),  # mid: mid channel
        )
        imagen = ElucidatedImagen(
            unets=(unet1),
            image_sizes=cfg.dataset.dimensions(),
            image_width=cfg.dataset.dimensions()[0],
            channels=cfg.dataset.solution_dimension,   # Han Gao add the input to this args explicity
            random_crop_sizes=None,
            num_sample_steps=20,  # diff_args.num_sample_steps, # original is 10
            cond_drop_prob=0.1,
            sigma_min=0.002,
            sigma_max=(80),      # max noise level, double the max noise level for upsampler  （80，160）
            sigma_data=0.5,      # standard deviation of data distribution
            rho=7,               # controls the sampling schedule
            P_mean=-1.2,         # mean of log-normal distribution from which noise is drawn for training
            P_std=1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn=80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
            condition_on_text=False,
            auto_normalize_img=False  # Han Gao make it false
        )
        imagen_trainer = ImagenTrainer(imagen, device=torch.device(cfg.device))

    downsampler = nn.Upsample(size=cfg.dataset.coarse_dimensions(), mode=cfg.dataset.upsample_mode)
    upsampler = nn.Upsample(size=cfg.dataset.dimensions(), mode=cfg.dataset.upsample_mode)
    train_upsampler = TrainUpsampler(cfg, downsampler, upsampler, imagen_trainer)

    logger = loggers.CSVLogger(cfg.run_dir, name=None)

    cbs = [
        callbacks.TimeStepProgressBar(cfg),
        callbacks.LogStats(),
        callbacks.ModelCheckpoint(
            dirpath=cfg.run_dir,
            filename='{epoch}',
            save_last='link',
            monitor='train_loss',
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
        max_epochs=cfg.model.epoch_count,
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=cbs,
        # fast_dev_run=2,
        # profiler='simple',
    )

    trainer.fit(train_upsampler, datamodule=dataset)


if __name__ == '__main__':
    last_override, run_dir = utils.get_run_dir()
    utils.set_run_dir(last_override, run_dir)
    main()
