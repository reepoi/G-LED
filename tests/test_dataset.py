import pytest
from torch.utils.data import DataLoader

from fixtures import init_hydra_cfg, engine

from conf import conf
from g_led.data import data_bfs_preprocess
from g_led import dataset


def test_backward_facing_step_old_and_new_datasets_equal(engine):
    cfg = init_hydra_cfg('conf', ['dataset=BackwardFacingStep2D', 'dataset.batch_size_val=17', 'model=TransformerBackwardFacingStep2D'])
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, cfg)
        dl = dataset.BackwardFacingStep2D(cfg.dataset)
        dl.prepare_data()
        dl.setup('validate')
        dset = data_bfs_preprocess.bfs_dataset(
            cfg.dataset.data_dir,
            trajec_max_len=cfg.dataset.trajectory_time_step_count_val,
            start_n=cfg.dataset.time_step_count_train,
            n_span=cfg.dataset.time_step_count_val
        )
        assert (dset.solution.unfold(0, cfg.dataset.trajectory_time_step_count_val, 1).permute(0, 4, 1, 2, 3) == dl.val).all()
        dloader = DataLoader(dataset=dset, batch_size=cfg.dataset.batch_size_val, shuffle=False)
        for i, (batch_original, batch) in enumerate(zip(dloader, dl.val_dataloader(shuffle=False))):
            print(i)
            assert (batch_original == batch[0]['val'][:batch_original.shape[0]]).all()
