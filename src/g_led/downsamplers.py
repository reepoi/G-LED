import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
import torch.nn as nn

from conf import downsampler


class Downsampler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch, flatten_solution=False):
        batch_size, time_count = batch.shape[:2]
        batch_macro = self.micro_to_macro(
            batch.view(-1, self.cfg.solution_dimension, *self.cfg.dimensions())
        ).view(batch_size, time_count, self.cfg.solution_dimension, *self.cfg.coarse_dimensions())
        if flatten_solution:
            batch_macro = batch_macro.view(batch_size, time_count, self.cfg.solution_dimension * self.cfg.embedding_dimension)
        return batch_macro


class DownsamplerLinear(Downsampler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.downsampler = nn.Upsample(
            size=cfg.coarse_dimensions(),
            mode=downsampler.DownsamplerLinearMode(len(cfg.dimensions())).name.lower(),
        )

    def micro_to_macro(self, batch_micro):
        return self.downsampler(batch_micro)


class DownsamplerGaussian(Downsampler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scale = 16
        self.kernel_width = 33
        self.padding = [16] * 4
        kernel_size = 33
        self.downsampler = getattr(nn, f'Conv{len(self.cfg.dimensions())}d')(
            in_channels=self.cfg.solution_dimension,
            out_channels=self.cfg.solution_dimension,
            groups=self.cfg.solution_dimension,
            kernel_size=kernel_size,
            bias=False,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            padding_mode='replicate',
        )
        spatial_dims = self.downsampler.weight.shape[2:]
        self.downsampler.weight = nn.Parameter(
            self.init_gaussian_kernel(spatial_dims, 0.4 * (kernel_size // 2)).expand(
                self.downsampler.out_channels, 1, *spatial_dims
            ).clone(),
            requires_grad=False,
        )

    def micro_to_macro(self, batch_micro):
        return self.downsampler(batch_micro)

    @staticmethod
    def init_gaussian_kernel(kernel_size, sigma, device=None, dtype=torch.float32):
        kernel = np.zeros(kernel_size)
        # set element at the middle to one, a dirac delta
        kernel[tuple(s//2 for s in kernel_size)] = 1.
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        kernel = torch.from_numpy(scipy.ndimage.gaussian_filter(kernel, sigma))
        return kernel[[None] * kernel.ndim].to(device=device, dtype=dtype)


class Upsampler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch):
        batch_size, time_count = batch.shape[:2]
        batch_micro = self.macro_to_micro(
            batch.view(-1, self.cfg.solution_dimension, *self.cfg.coarse_dimensions())
        ).view(batch_size, time_count, self.cfg.solution_dimension, *self.cfg.dimensions())
        return batch_micro


class UpsamplerLinear(Upsampler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.upsampler = nn.Upsample(
            size=cfg.dimensions(),
            mode=downsampler.DownsamplerLinearMode(len(cfg.dimensions())).name.lower(),
        )

    def macro_to_micro(self, batch_micro):
        return self.upsampler(batch_micro)


def get_downsampler(cfg):
    match cfg.downsampler:
        case downsampler.DownsamplerLinear():
            return DownsamplerLinear(cfg)
        case downsampler.DownsamplerGaussian():
            return DownsamplerGaussian(cfg)
        case _:
            raise ValueError(f'Unknown downsampler: {cfg.downsampler}')


def get_upsampler(cfg):
    match cfg.upsampler:
        case downsampler.UpsamplerLinear():
            return UpsamplerLinear(cfg)
        case _:
            raise ValueError(f'Unknown upsampler: {cfg.upsampler}')
