import enum

import omegaconf
from hydra_orm import orm
import sqlalchemy as sa


class Downsampler(orm.InheritableTable):
    pass


class DownsamplerLinearMode(enum.IntEnum):
    LINEAR = 1
    BILINEAR = 2
    TRILINEAR = 3


class DownsamplerLinear(Downsampler):
    pass


class DownsamplerGaussian(Downsampler):
    pass


class Upsampler(orm.InheritableTable):
    pass


class UpsamplerLinear(Upsampler):
    pass
