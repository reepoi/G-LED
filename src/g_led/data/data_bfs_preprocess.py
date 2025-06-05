from pathlib import Path
import numpy as np
import pdb
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import lightning.pytorch as pl
from pytorch_lightning.utilities import CombinedLoader


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

    def setup(self, stage):
        if stage == 'fit':
            solution = torch.load(self.data_dir/'data_cat.pt')
            self.train = (
                solution[self.solution_start_times['train']:self.solution_end_times['train']]
                .unfold(0, self.trajectory_max_lens['train'], 1)
                .permute(0, 4, 1, 2, 3)
            )
            self.val_on_train = (
                solution[self.solution_start_times['train']:self.solution_end_times['train']]
                .unfold(0, self.trajectory_max_lens['val'], 1)
                .permute(0, 4, 1, 2, 3)
            )
            self.val = (
                solution[self.solution_start_times['val']:self.solution_end_times['val']]
                .unfold(0, self.trajectory_max_lens['val'], 1)
                .permute(0, 4, 1, 2, 3)
            )
        elif stage == 'validate':
            solution = torch.load(self.data_dir/'data_cat.pt')
            self.val_on_train = (
                solution[self.solution_start_times['train']:self.solution_end_times['train']]
                .unfold(0, self.trajectory_max_lens['val'], 1)
                .permute(0, 4, 1, 2, 3)
            )
            self.val = (
                solution[self.solution_start_times['val']:self.solution_end_times['val']]
                .unfold(0, self.trajectory_max_lens['val'], 1)
                .permute(0, 4, 1, 2, 3)
            )
        elif stage == 'test':
            raise NotImplementedError()
        elif stage == 'predict':
            solution = torch.load(self.data_dir/'data_cat.pt')
            self.val_on_train = (
                solution[self.solution_start_times['train']:self.solution_end_times['train']]
                .unfold(0, self.trajectory_max_lens['val'], 1)
                .permute(0, 4, 1, 2, 3)
            )
            self.val = (
                solution[self.solution_start_times['val']:self.solution_end_times['val']]
                .unfold(0, self.trajectory_max_lens['val'], 1)
                .permute(0, 4, 1, 2, 3)
            )
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
    def __init__(self,
                 data_location=['/root/workspace/out/diffusion-dynamics/G-LED/data/data0.npy',
                                '/root/workspace/out/diffusion-dynamics/G-LED/data/data1.npy'],
                 trajec_max_len=50,
                 start_n=0,
                 n_span=510):
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
        solution = torch.load('/root/workspace/out/diffusion-dynamics/G-LED/data/data_cat.pt')
        self.solution = solution[start_n:start_n+n_span]

    def __len__(self):
        return self.n_span - self.trajec_max_len

    def __getitem__(self, index):
        item = self.solution[index:index+self.trajec_max_len]
        return item


if __name__ == '__main__':
    dl = BackwardFacingStep(
        Path('/root/workspace/out/diffusion-dynamics/G-LED/data'),
        trajectory_max_lens=dict(train=50, val=50),
        solution_start_times=dict(train=0, val=0),
        solution_end_times=dict(train=510, val=510),
        batch_sizes=dict(train=20, val=20)
    )
    dl.prepare_data()
    dl.setup('validate')
    dset = bfs_dataset()
    dloader = DataLoader(dataset=dset, batch_size=20, shuffle=False)
    for i, (batch_original, batch) in enumerate(zip(dloader, dl.val_dataloader(shuffle=False))):
        print(i)
        if not (batch_original == batch[0]['val']).all():
            breakpoint()
            print('Do something!')
