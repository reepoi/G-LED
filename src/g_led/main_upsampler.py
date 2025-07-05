from collections import defaultdict
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import torch
import lightning.pytorch as pl
from einops import rearrange

from conf import conf
from g_led import callbacks, datasets, models, loggers, utils


log = utils.getLoggerByFilename(__file__)


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
        model = models.get_model(cfg)

    logger = loggers.CSVLogger(cfg.run_dir, name=None)

    cbs = [
        callbacks.TimeStepProgressBar(cfg),
        callbacks.LogStats(),
        callbacks.ModelCheckpoint(
            dirpath=cfg.run_dir,
            filename='{epoch}',
            save_last=None,
            # monitor='loss',
            # save_top_k=2,
            every_n_train_steps=500,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        )
    ]
    trainer = pl.Trainer(
        # detect_anomaly=True,
        # strategy='ddp',
        accelerator=cfg.device,
        # devices=4,
        devices=1,
        logger=logger,
        max_epochs=cfg.get_model().epoch_count,
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=cbs,
        # log_every_n_steps=5,
        # fast_dev_run=2,
        # profiler='simple',
    )

    if cfg.fit:
        trainer.fit(model, datamodule=dataset)
    if cfg.predict:
        batches_micro = trainer.predict(model, datamodule=dataset)
        trajectory_count = cfg.dataset.trajectory_count
        if cfg.dataset.trajectories_are_shared_across_splits:
            trajectory_count //= 3  # splits are train, val, and test
        batches_micro = rearrange(
            torch.cat(batches_micro),
            '(trajectory trajectory_window) time_step ... -> trajectory (trajectory_window time_step) ...',
            trajectory=trajectory_count,
        ).cpu()
        torch.save(batches_micro, cfg.run_dir/'pred.pt')


if __name__ == '__main__':
    last_override, run_dir = utils.get_run_dir()
    utils.set_run_dir(last_override, run_dir)
    main()
