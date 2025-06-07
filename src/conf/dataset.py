from dataclasses import field
from typing import List, Any
from pathlib import Path

from omegaconf import II
import hydra_orm.utils
from hydra_orm import orm
import sqlalchemy as sa


class Dataset(orm.InheritableTable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        '_self_',
    ])
    _data_dir: str = field(default=str(Path('/mnta/taosData/diffusion-dynamics/G-LED/data').resolve()))
    time_step_count_train: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    time_step_count_val: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    time_step_count_test: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)

    trajectory_time_step_count_train: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    trajectory_time_step_count_val: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    trajectory_time_step_count_test: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=II('.trajectory_time_step_count_val'))

    batch_size_train: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    batch_size_val: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    batch_size_test: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=II('.batch_size_val'))

    @property
    def data_dir(self):
        return Path(self._data_dir)


class KuramotoSivashinsky1D(Dataset):
    dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1024)
    coarse_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=16)

    def dimensions(self):
        return [self.dimension]

    def coarse_dimensions(self):
        return [self.coarse_dimension]

    @property
    def embedding_dimension(self):
        return self.coarse_dimension


class BackwardFacingStep2D(Dataset):
    dimension_height: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    dimension_width: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    coarse_dimension_height: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=32)
    coarse_dimension_width: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=32)

    def dimensions(self):
        return [self.dimension_height, self.dimension_width]

    def coarse_dimensions(self):
        return [self.coarse_dimension_height, self.coarse_dimension_width]

    @property
    def embedding_dimension(self):
        return self.coarse_dimension_height * self.coarse_dimension_width


class ChannelFlow3D(Dataset):
    dimension_height: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    dimension_length: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    dimension_width: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    coarse_dimension_height: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=8)
    coarse_dimension_length: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=32)
    coarse_dimension_width: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=8)

    def dimensions(self):
        return [self.dimension_height, self.dimension_length, self.dimension_width]

    def coarse_dimensions(self):
        return [self.coarse_dimension_height, self.coarse_dimension_length, self.coarse_dimension_width]

    @property
    def embedding_dimension(self):
        return self.coarse_dimension_height * self.coarse_dimension_length * self.coarse_dimension_width
