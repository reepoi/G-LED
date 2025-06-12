import pprint

import lightning.pytorch as pl
import numpy as np
import torch
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, IterableDataset
import hydra
from omegaconf import OmegaConf
import dapper.mods.KS
from einops import rearrange, EinopsError
from tqdm import tqdm

from conf import conf, dataset
from g_led import utils


log = utils.getLoggerByFilename(__file__)


class GeneratorDataset(IterableDataset):
    def __init__(self, iterable):
        super().__init__()
        self.iterable = iterable

    def __len__(self):
        return len(self.iterable)

    def __iter__(self):
        for item in self.iterable:
            yield item


class TrajectoryDataset(pl.lightning.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        raise NotImplementedError()

    def load_trajectories(self):
        raise NotImplementedError()

    def extract_from_trajectories(self, trajectories, start, end, trajectory_time_step_count):
        raise NotImplementedError()

    def get_train_split(self, trajectories, trajectory_time_step_count=None):
        if trajectory_time_step_count is None:
            trajectory_time_step_count = self.cfg.time_step_window_size_train
        return self.extract_from_trajectories(
            trajectories[self.cfg.trajectory_start_train:self.cfg.trajectory_end_train],
            0, self.cfg.macro_time_step_count_train,
            trajectory_time_step_count
        )

    def get_val_split(self, trajectories):
        return self.extract_from_trajectories(
            trajectories[self.cfg.trajectory_start_val:self.cfg.trajectory_end_val],
            self.cfg.macro_time_step_count_train, self.cfg.macro_time_step_end_val,
            self.cfg.time_step_window_size_val
        )

    def get_test_split(self, trajectories):
        return self.extract_from_trajectories(
            trajectories[self.cfg.trajectory_start_test:self.cfg.trajectory_end_test],
            self.cfg.macro_time_step_end_val, self.cfg.macro_time_step_end_test,
            self.cfg.time_step_window_size_test
        )

    def setup(self, stage):
        trajectories = self.load_trajectories()
        self.validate_trajectories(trajectories)
        if stage == 'fit':
            self.train = self.get_train_split(trajectories)
            self.val_on_train = self.get_train_split(trajectories, self.cfg.time_step_window_size_val)
            self.val = self.get_val_split(trajectories)
            self.validate_splits(['train', 'val_on_train', 'val'])
        elif stage == 'validate':
            self.val_on_train = self.get_train_split(trajectories, self.cfg.time_step_window_size_val)
            self.val = self.get_val_split(trajectories)
            self.validate_splits(['val_on_train', 'val'])
        elif stage == 'test':
            self.test = self.get_test_split(trajectories)
        elif stage == 'predict':
            self.val_on_train = self.get_train_split(trajectories, self.cfg.time_step_window_size_val)
            self.val = self.get_val_split(trajectories)
            self.test = self.get_test_split(trajectories)
            self.validate_splits(['val_on_train', 'val', 'test'])
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def validate_trajectories(self, trajectories):
        space_dims = dict(zip(['width', 'length', 'height'], self.cfg.dimensions()))
        axis_names = f"trajectory time component {' '.join(space_dims)}"
        trajectory_count = self.cfg.trajectory_count
        if self.cfg.trajectories_are_shared_across_splits:
            trajectory_count //= 3  # splits are train, val, and test
        try:
            rearrange(
                trajectories,
                f'{axis_names} -> {axis_names}',
                trajectory=trajectory_count,
                time=self.cfg.trajectory_time_step_count_macro,
                component=self.cfg.solution_dimension,
                **space_dims,
            )
        except EinopsError as e:
            log.critical(
                "Trajectories saved at '%s' do not have the expected shape. See the following einops.EinopsError for details:\n%s",
                self.cfg.data_dir/self.cfg.processed_filename, e
            )
            raise RuntimeError('Trajectories did not pass validation. See the logs for more details.')

    def validate_splits(self, splits):
        space_dims = dict(zip(['width', 'length', 'height'], self.cfg.dimensions()))
        axis_names = f"trajectory time component {' '.join(space_dims)}"
        split_to_cfg_field_trajectory_count = dict(train='train', val_on_train='train', val='val', test='test')
        split_to_cfg_field_time_step_window_size = dict(train='train', val_on_train='val', val='val', test='test')
        has_validation_error = False
        for split in splits:
            trajectory_count = getattr(self.cfg, f'trajectory_count_{split_to_cfg_field_trajectory_count[split]}')
            trajectory_time_step_count_macro = getattr(self.cfg, f'macro_time_step_count_{split_to_cfg_field_trajectory_count[split]}') or self.cfg.trajectory_time_step_count_macro
            if (time_step_window_size := getattr(self.cfg, f'time_step_window_size_{split_to_cfg_field_time_step_window_size[split]}')) is not None:
                # assuming window stride of 1
                trajectory_count *= trajectory_time_step_count_macro - time_step_window_size + 1
                trajectory_time_step_count_macro = time_step_window_size
            try:
                rearrange(
                    getattr(self, split),
                    f'{axis_names} -> {axis_names}',
                    trajectory=trajectory_count,
                    time=trajectory_time_step_count_macro,
                    component=self.cfg.solution_dimension,
                    **space_dims,
                )
            except EinopsError as e:
                has_validation_error = True
                log.critical(
                    "The '%s' split of the trajectories saved at '%s' does not have the expected shape. See the following einops.EinopsError for details:\n%s",
                    split, self.cfg.data_dir/self.cfg.processed_filename, e
                )
        if has_validation_error:
            raise RuntimeError('The trajectory splits did not pass validation. See the logs for more details.')

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


class KuramotoSivashinksy1D(TrajectoryDataset):
    def prepare_data(self):
        if (self.cfg.data_dir/self.cfg.processed_filename).exists():
            return

        device = 'cpu'
        domain_width = self.cfg.domain_width
        if self.cfg.domain_width_is_multiple_of_pi:
            domain_width *= torch.pi
        solver = self.etd_rk4_wrapper(self.cfg, 'cpu', domain_width)

        trajectories = []
        for _ in tqdm(range(self.cfg.trajectory_count_train + self.cfg.trajectory_count_val + self.cfg.trajectory_count_test), desc='Computing trajectories'):
            x0 = torch.tensor(dapper.mods.KS.Model(
                dt=self.cfg.trajectory_time_step_size_micro,
                DL=domain_width,
                Nx=self.cfg.dimension,
            ).x0, device=device, dtype=torch.float32)[None]
            state = x0 + torch.randn_like(x0)

            solution = [state]
            for _ in range(self.cfg.trajectory_time_step_count_micro):
                state = solver(None, state)
                solution.append(state)
            solution = solution[::self.cfg.trajectory_time_step_subsample_interval_macro]
            trajectories.append(rearrange(solution, 'time 1 space -> time 1 space'))

        trajectories = rearrange(trajectories, 'trajectory time 1 space -> trajectory time 1 space')
        torch.save(trajectories, self.cfg.data_dir/self.cfg.processed_filename)

    def load_trajectories(self):
        return torch.load(self.cfg.data_dir/self.cfg.processed_filename)

    def extract_from_trajectories(self, trajectories, start, end, time_step_window_size):
        if time_step_window_size is None:
            time_step_window_size = trajectories.shape[1]
        return rearrange(
            trajectories[:, start:end]
            .unfold(1, time_step_window_size, 1),
            'trajectory trajectory_window component space time -> (trajectory trajectory_window) time component space'
        )

    def etd_rk4_wrapper(self, cfg, device, domain_width):
        """ Returns an ETD-RK4 integrator for the KS equation. Currently very specific, need
        to adjust this to fit into the same framework as the ODE integrators

        Directly ported from https://github.com/nansencenter/DAPPER/blob/master/dapper/mods/KS/core.py
        which is adapted from kursiv.m of Kassam and Trefethen, 2002, doi.org/10.1137/S1064827502410633.
        """
        dtype = torch.float32
        kk = np.append(np.arange(0, cfg.dimension / 2), 0) * 2 / domain_width  # wave nums for rfft
        h = cfg.trajectory_time_step_size_micro

        # Operators
        L = kk ** 2 - kk ** 4  # Linear operator for K-S eqn: F[ - u_xx - u_xxxx]

        # Precompute ETDRK4 scalar quantities
        E = torch.tensor(np.exp(h * L), device=device, dtype=dtype).unsqueeze(0)  # Integrating factor, eval at dt
        E2 = torch.tensor(np.exp(h * L / 2), device=device, dtype=dtype).unsqueeze(0)  # Integrating factor, eval at dt/2

        # Roots of unity are used to discretize a circular contour...
        nRoots = 16
        roots = np.exp(1j * np.pi * (0.5 + np.arange(nRoots)) / nRoots)
        # ... the associated integral then reduces to the mean,
        # g(CL).mean(axis=-1) ~= g(L), whose computation is more stable.
        CL = h * L[:, None] + roots  # Contour for (each element of) L
        # E * exact_integral of integrating factor:
        Q = torch.tensor(h * ((np.exp(CL / 2) - 1) / CL).mean(axis=-1).real, dtype=dtype, device=device).unsqueeze(0)
        # RK4 coefficients (modified by Cox-Matthews):
        f1 = torch.tensor(h * ((-4 - CL + np.exp(CL) * (4 - 3 * CL + CL ** 2)) / CL ** 3).mean(axis=-1).real, dtype=dtype, device=device).unsqueeze(0)
        f2 = torch.tensor(h * ((2 + CL + np.exp(CL) * (-2 + CL)) / CL ** 3).mean(axis=-1).real, dtype=dtype, device=device).unsqueeze(0)
        f3 = torch.tensor(h * ((-4 - 3 * CL - CL ** 2 + np.exp(CL) * (4 - CL)) / CL ** 3).mean(axis=-1).real, dtype=dtype, device=device).unsqueeze(0)

        D = 1j * torch.tensor(kk, device=device, dtype=dtype)  # Differentiation to compute:  F[ u_x ]

        def NL(v):
            return -.5 * D * torch.fft.rfft(torch.fft.irfft(v, dim=-1) ** 2, dim=-1)

        def inner(t, v):
            v = torch.fft.rfft(v, dim=-1)
            N1 = NL(v)
            v1 = E2 * v + Q * N1

            N2a = NL(v1)
            v2a = E2 * v + Q * N2a

            N2b = NL(v2a)
            v2b = E2 * v1 + Q * (2 * N2b - N1)

            N3 = NL(v2b)
            v = E * v + N1 * f1 + 2 * (N2a + N2b) * f2 + N3 * f3
            return torch.fft.irfft(v, dim=-1)

        return inner


class BackwardFacingStep2D(TrajectoryDataset):
    def prepare_data(self):
        if (self.cfg.data_dir/self.cfg.processed_filename).exists():
            return
        if (self.cfg.data_dir/'data_cat_f32.pt').exists():
            torch.save(
                torch.load(self.cfg.data_dir/'data_cat_f32.pt')[None],
                self.cfg.data_dir/self.cfg.processed_filename
            )
            return

        trajectories0 = np.load(self.cfg.data_dir/'data0.npy', allow_pickle=True)
        trajectories1 = np.load(self.cfg.data_dir/'data1.npy', allow_pickle=True)
        trajectories  = np.concatenate((trajectories0, trajectories1), axis=0)
        trajectories = torch.from_numpy(trajectories)
        torch.save(trajectories[None].to(torch.float32), self.cfg.data_dir/self.cfg.processed_filename)

    def load_trajectories(self):
        return torch.load(self.cfg.data_dir/self.cfg.processed_filename)

    def extract_from_trajectories(self, trajectories, start, end, time_step_window_size):
        if time_step_window_size is None:
            time_step_window_size = trajectories.shape[1]
        return rearrange(
            trajectories[:, start:end]
            .unfold(1, time_step_window_size, 1),
            'trajectory trajectory_window component width length time -> (trajectory trajectory_window) time component width length'
        )


def get_dataset(cfg):
    if isinstance(cfg, dataset.KuramotoSivashinsky1D):
        return KuramotoSivashinksy1D(cfg)
    elif isinstance(cfg, dataset.BackwardFacingStep2D):
        return BackwardFacingStep2D(cfg)
    elif isinstance(cfg, dataset.ChannelFlow3D):
        raise NotImplementedError()
    else:
        raise ValueError(f'Unknown dataset: {cfg}')


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        pprint.pp(cfg)
        pl.seed_everything(cfg.rng_seed)
        with pl.utilities.seed.isolate_rng():
            dataset = get_dataset(cfg.dataset)
            dataset.prepare_data()
        dataset.setup('fit')
        breakpoint()
        print('end')


if __name__ == '__main__':
    main()
