import pytest
from torch.utils.data import DataLoader

from fixtures import init_hydra_cfg, engine

from conf import conf
from g_led.data import data_bfs_preprocess


def test_backward_facing_step_old_and_new_datasets_equal(engine):
    cfg = init_hydra_cfg('conf', ['dataset=Dataset'])
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, cfg)
        dl = data_bfs_preprocess.BackwardFacingStep(
            cfg.data_dir,
            trajectory_max_lens=dict(train=50, val=50),
            solution_start_times=dict(train=0, val=0),
            solution_end_times=dict(train=510, val=510),
            batch_sizes=dict(train=20, val=20)
        )
        dl.prepare_data()
        dl.setup('validate')
        dset = data_bfs_preprocess.bfs_dataset(cfg.data_dir)
        dloader = DataLoader(dataset=dset, batch_size=20, shuffle=False)
        for i, (batch_original, batch) in enumerate(zip(dloader, dl.val_dataloader(shuffle=False))):
            print(i)
            assert (batch_original == batch[0]['val']).all()
