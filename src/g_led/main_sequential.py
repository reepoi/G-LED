from collections import defaultdict
import pprint
import sys

from einops import reduce
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import lightning.pytorch as pl

from conf import conf
from g_led import callbacks, datasets, models, loggers, utils
from g_led.transformer.sequentialModel import SequentialModel as Transformer


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

    # time_step_time_logger = loggers.CSVLogger(cfg.run_dir, name=None, name_metrics_file='time_step_times.csv')

    pl.seed_everything(cfg.rng_seed)
    with pl.utilities.seed.isolate_rng():
        dataset = datasets.get_dataset(cfg.dataset)
        dataset.prepare_data()
    with pl.utilities.seed.isolate_rng():
        model = models.get_model(cfg)

    logger = loggers.CSVLogger(cfg.run_dir, name=None)

    cbs = [
        callbacks.TimeStepProgressBar(cfg),
        callbacks.LogStatsSequential(),  # callbacks.LogStatsSequential saves the checkpoints
        # callbacks.ModelCheckpoint(
        #     dirpath=cfg.run_dir,
        #     filename='{epoch}__{forecast_time_step_count:.0f}',
        #     save_last='link',
        #     monitor='ckpt_metric',
        #     mode='min',
        #     save_top_k=2,
        #     save_on_train_epoch_end=False,
        #     enable_version_counter=False,
        # ),
    ]
    trainer = pl.Trainer(
        # detect_anomaly=True,
        # strategy='ddp',
        strategy='ddp_find_unused_parameters_true',
        accelerator=cfg.device,
        devices=3,
        # devices=1,
        logger=logger,
        max_epochs=cfg.get_model().epoch_count,
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=cbs,
        enable_checkpointing=False,  # callbacks.LogStatsSequential saves the checkpoints
        # fast_dev_run=2,
        # profiler='simple',
    )

    if cfg.fit:
        log.info(
            'torchinfo:\n%s',
            torchinfo.summary(
                model.model,
                input_data=torch.ones(
                    cfg.dataset.batch_size_train,
                    cfg.get_model().time_step_window_size,
                    cfg.dataset.solution_dimension * cfg.dataset.embedding_dimension
                )
            )
        )
        trainer.fit(model, datamodule=dataset)
    # if cfg.predict:
    #     prediction_batches = trainer.predict(model, datamodule=dataset)
    #     predictions = defaultdict(list)
    #     for batch in prediction_batches:
    #         for k, v in batch.items():
    #             predictions[k].append(v)
    #     for k, v in list(predictions.items()):
    #         v = torch.cat(v).cpu()
    #         torch.save(v, cfg.run_dir/f'pred_{k}.pt')
    #         del predictions[k]


if __name__ == '__main__':
    last_override, run_dir = utils.get_run_dir()
    utils.set_run_dir(last_override, run_dir)
    main()
