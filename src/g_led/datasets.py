import pprint

import lightning.pytorch as pl
import numpy as np
import torch
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, IterableDataset
import hydra
from omegaconf import OmegaConf

from conf import conf
import conf.dataset
from g_led import utils


class GeneratorDataset(IterableDataset):
    def __init__(self, iterable):
        super().__init__()
        self.iterable = iterable

    def __len__(self):
        return len(self.iterable)

    def __iter__(self):
        for item in self.iterable:
            yield item


class KuramotoSivashinksy(pl.lightning.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


class BackwardFacingStep2D(pl.lightning.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        assert (self.cfg.data_dir/'data0.npy').exists()
        assert (self.cfg.data_dir/'data1.npy').exists()
        if (self.cfg.data_dir/'data_cat_f32.pt').exists():
            return

        solution0 = np.load(self.cfg.data_dir/'data0.npy', allow_pickle=True)
        solution1 = np.load(self.cfg.data_dir/'data1.npy', allow_pickle=True)
        solution  = np.concatenate((solution0, solution1), axis=0)
        solution = torch.from_numpy(solution)
        torch.save(solution.to(torch.float32), self.cfg.data_dir/'data_cat_f32.pt')

    def extract_from_solution(self, solution, start, end, trajectory_time_step_count):
        return (
            solution[start:end]
            .unfold(0, trajectory_time_step_count, 1)
            .permute(0, 4, 1, 2, 3)
        )

    def get_train_split(self, solution, trajectory_time_step_count=None):
        if trajectory_time_step_count is None:
            trajectory_time_step_count = self.cfg.trajectory_time_step_count_train
        return self.extract_from_solution(solution, 0, self.cfg.time_step_count_train, trajectory_time_step_count)

    def get_val_split(self, solution):
        return self.extract_from_solution(
            solution,
            self.cfg.time_step_count_train, self.cfg.time_step_count_train + self.cfg.time_step_count_val,
            self.cfg.trajectory_time_step_count_val
        )

    def get_test_split(self, solution):
        return self.extract_from_solution(
            solution,
            self.cfg.time_step_count_train + self.cfg.time_step_count_val,
            self.cfg.time_step_count_train + self.cfg.time_step_count_val + self.cfg.time_step_count_test,
            self.cfg.trajectory_time_step_count_test
        )

    def setup(self, stage):
        solution = torch.load(self.cfg.data_dir/'data_cat_f32.pt')
        if stage == 'fit':
            self.train = self.get_train_split(solution)
            self.val_on_train = self.get_train_split(solution, self.cfg.trajectory_time_step_count_val)
            self.val = self.get_val_split(solution)
        elif stage == 'validate':
            self.val_on_train = self.get_train_split(solution, self.cfg.trajectory_time_step_count_val)
            self.val = self.get_val_split(solution)
        elif stage == 'test':
            self.test = self.get_test_split(solution)
        elif stage == 'predict':
            self.val_on_train = self.get_train_split(solution, self.cfg.trajectory_time_step_count_val)
            self.val = self.get_val_split(solution)
            self.test = self.get_test_split(solution)
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train, shuffle=shuffle, batch_size=self.cfg.batch_size_train)

    def val_dataloader(self, shuffle=True, combined=True, split_limits=None):
        val_on_train = DataLoader(self.val_on_train, shuffle=shuffle, batch_size=self.cfg.batch_size_val)
        val = DataLoader(self.val, shuffle=shuffle, batch_size=self.cfg.batch_size_val)
        dataloaders = dict(val_on_train=val_on_train, val=val)

        if split_limits is None:
            split_limits = dict(val_on_train=5, val=2)
        for split, dataloader in dataloaders.items():
            limit = split_limits[split]
            if limit is not None:
                dataloaders[split] = DataLoader(GeneratorDataset([batch for _, batch in zip(range(limit), dataloader)]), collate_fn=lambda x: x[0])

        if combined:
            return CombinedLoader(dataloaders, mode='sequential')

        return dataloaders

    def test_dataloader(self, shuffle=False):
        return DataLoader(self.test, shuffle=shuffle, batch_size=self.cfg.batch_size_test)

    def predict_dataloader(self, shuffle=False, combined=True, val_split_limits=None):
        dataloaders = self.val_dataloader(shuffle=shuffle, combined=False, split_limits=val_split_limits)
        dataloaders['test'] = self.test_dataloader(shuffle=shuffle)

        if combined:
            return CombinedLoader(dataloaders, mode='sequential')

        return dataloaders


def get_dataset(cfg):
    if isinstance(cfg, conf.dataset.KuramotoSivashinsky1D):
        raise NotImplementedError()
    elif isinstance(cfg, conf.dataset.BackwardFacingStep2D):
        return BackwardFacingStep2D(cfg)
    elif isinstance(cfg, conf.dataset.ChannelFlow3D):
        raise NotImplementedError()
    else:
        raise ValueError(f'Unknown dataset: {cfg}')


@hydra.main(**{**utils.HYDRA_INIT, 'config_path': '../../../conf'})
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        pprint.pp(cfg)
        print('end')


if __name__ == '__main__':
    main()
