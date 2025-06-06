from pathlib import Path
import pprint

import lightning.pytorch as pl
import numpy as np
import torch
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Dataset
import hydra
from omegaconf import OmegaConf

from conf import conf
from g_led import utils


class BackwardFacingStep(pl.lightning.LightningDataModule):
    def __init__(self, data_dir, trajectory_max_lens, solution_start_times, solution_end_times, batch_sizes):
        self.data_dir = data_dir
        self.trajectory_max_lens = trajectory_max_lens
        self.solution_start_times = solution_start_times
        self.solution_end_times = solution_end_times
        self.batch_sizes = batch_sizes

    def prepare_data(self):
        assert (self.data_dir/'data0.npy').exists()
        assert (self.data_dir/'data1.npy').exists()
        if (self.data_dir/'data_cat.pt').exists():
            return

        solution0 = np.load(self.data_dir/'data0.npy', allow_pickle=True)
        solution1 = np.load(self.data_dir/'data1.npy', allow_pickle=True)
        solution  = np.concatenate((solution0, solution1), axis=0)
        solution = torch.from_numpy(solution)
        torch.save(solution, self.data_dir/'data_cat.pt')

    def extract_split_from_solution(self, solution, times_split, len_split=None):
        len_split = len_split or times_split
        return (
            solution[self.solution_start_times[times_split]:self.solution_end_times[times_split]]
            .unfold(0, self.trajectory_max_lens[len_split], 1)
            .permute(0, 4, 1, 2, 3)
        )

    def setup(self, stage):
        if stage == 'fit':
            solution = torch.load(self.data_dir/'data_cat.pt')
            self.train = self.extract_split_from_solution(solution, 'train')
            self.val_on_train = self.extract_split_from_solution(solution, 'train', 'val')
            self.val = self.extract_split_from_solution(solution, 'val')
        elif stage == 'validate':
            solution = torch.load(self.data_dir/'data_cat.pt')
            self.val_on_train = self.extract_split_from_solution(solution, 'train', 'val')
            self.val = self.extract_split_from_solution(solution, 'val')
        elif stage == 'test':
            raise NotImplementedError()
        elif stage == 'predict':
            solution = torch.load(self.data_dir/'data_cat.pt')
            self.val_on_train = self.extract_split_from_solution(solution, 'train', 'val')
            self.val = self.extract_split_from_solution(solution, 'val')
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_sizes['train'])

    def val_dataloader(self, shuffle=True, combined=True):
        val_on_train = DataLoader(self.val_on_train, shuffle=shuffle, batch_size=self.batch_sizes['val'])
        val = DataLoader(self.val, shuffle=shuffle, batch_size=self.batch_sizes['val'])
        if combined:
            return CombinedLoader(dict(val_on_train=val_on_train, val=val), mode='max_size')
        else:
            return dict(val_on_train=val_on_train, val=val)

    def test_dataloader(self):
        raise NotImplementedError()

    def predict_dataloader(self):
        return DataLoader(self.predict, shuffle=True, batch_size=self.batch_sizes['predict'])


class bfs_dataset(Dataset):
    def __init__(self, data_dir,
                 trajec_max_len=50,
                 start_n=0,
                 n_span=510):
        self.data_dir = data_dir
        assert n_span > trajec_max_len
        self.start_n = start_n
        self.n_span  = n_span
        self.trajec_max_len = trajec_max_len

        # print('load data0')
        # solution0 = np.load(data_location[0],allow_pickle = True)
        # print('load data1')
        # solution1 = np.load(data_location[1],allow_pickle = True)
        # print('cat')
        # solution  = np.concatenate([solution0,
        #                             solution1],axis = 0)
        # print('convert to torch')
        # self.solution = torch.from_numpy(solution[start_n:start_n+n_span])
        assert (self.data_dir/'data0.npy').exists()
        assert (self.data_dir/'data1.npy').exists()
        assert (self.data_dir/'data_cat.pt').exists()
        solution = torch.load(self.data_dir/'data_cat.pt')
        self.solution = solution[start_n:start_n+n_span]

    def __len__(self):
        return self.n_span - self.trajec_max_len

    def __getitem__(self, index):
        item = self.solution[index:index+self.trajec_max_len]
        return item


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
