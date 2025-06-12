from dataclasses import field
from typing import List, Any, Optional
from pathlib import Path

from omegaconf import II, SI
import hydra_orm.utils
from hydra_orm import orm
import sqlalchemy as sa


class Dataset(orm.InheritableTable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        '_self_',
    ])
    _data_dir: str = field(default=str(Path('/mnta/taosData/diffusion-dynamics/G-LED/data').resolve()))

    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=II('oc.select:..rng_seed,0'))
    _processed_filename: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)

    trajectory_count_train: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    trajectory_count_val: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    trajectory_count_test: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=II('.trajectory_count_val'))
    trajectories_are_shared_across_splits: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    trajectory_time_step_size_micro: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)
    # excluding the initial condition
    trajectory_time_step_count_micro: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)

    trajectory_time_step_subsample_interval_macro: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)

    # None implies use entire trajectory
    macro_time_step_count_train: Optional[int] = orm.make_field(sa.Column(sa.Integer), default=None)
    macro_time_step_count_val: Optional[int] = orm.make_field(sa.Column(sa.Integer), default=None)
    macro_time_step_count_test: Optional[int] = orm.make_field(sa.Column(sa.Integer), default=None)

    # None implies use window whose size is the entire trajectory
    time_step_window_size_train: Optional[int] = orm.make_field(sa.Column(sa.Integer), default=None)
    time_step_window_size_val: Optional[int] = orm.make_field(sa.Column(sa.Integer), default=None)
    time_step_window_size_test: Optional[int] = orm.make_field(sa.Column(sa.Integer), default=II('.time_step_window_size_val'))

    batch_size_train: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    batch_size_val: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    batch_size_test: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=II('.batch_size_val'))

    def __post_init__(self):
        if self.trajectories_are_shared_across_splits:
            if (
                self.trajectory_count_train != self.trajectory_count_val
                or
                self.trajectory_count_train != self.trajectory_count_test
            ):
                raise ValueError(
                    'Since dataset.trajectories_are_shared_across_splits=true,'
                    f' the settings dataset.trajectory_count_train={self.trajectory_count_train},'
                    f' dataset.trajectory_count_val={self.trajectory_count_val},'
                    f' and dataset.trajectory_count_test={self.trajectory_count_test}'
                    ' must be equal, but they are not.'
                )

    @property
    def data_dir(self):
        return Path(self._data_dir)

    @property
    def processed_filename(self):
        return f'{self._processed_filename}.pt'

    @property
    def trajectory_count(self):
        return self.trajectory_count_train + self.trajectory_count_val + self.trajectory_count_test

    @property
    def trajectory_time_step_count_macro(self):
        return (
            # add one to accomodate initial condition
            self.trajectory_time_step_count_micro + 1
        ) // self.trajectory_time_step_subsample_interval_macro + 1

    @property
    def trajectory_start_train(self):
        return 0

    @property
    def trajectory_end_train(self):
        return self.trajectory_count_train

    @property
    def trajectory_start_val(self):
        if self.trajectories_are_shared_across_splits:
            return self.trajectory_start_train
        else:
            return self.trajectory_end_train

    @property
    def trajectory_end_val(self):
        if self.trajectories_are_shared_across_splits:
            return self.trajectory_end_train
        elif self.trajectory_end_train is None or self.trajectory_count_val is None:
            return None
        else:
            return self.trajectory_end_train + self.trajectory_count_val

    @property
    def trajectory_start_test(self):
        if self.trajectories_are_shared_across_splits:
            return self.trajectory_start_train
        else:
            return self.trajectory_end_val

    @property
    def trajectory_end_test(self):
        if self.trajectories_are_shared_across_splits:
            return self.trajectory_end_train
        elif self.trajectory_end_val is None or self.trajectory_count_test is None:
            return None
        else:
            return self.trajectory_end_val + self.trajectory_count_test

    @property
    def macro_time_step_end_val(self):
        if self.macro_time_step_count_train is None or self.macro_time_step_count_val is None:
            macro_time_step_end = None
        else:
            macro_time_step_end = self.macro_time_step_count_train + self.macro_time_step_count_val
        return macro_time_step_end

    @property
    def macro_time_step_end_test(self):
        macro_time_step_end_val = self.macro_time_step_end_val()
        if macro_time_step_end_val is None or self.macro_time_step_count_test is None:
            macro_time_step_end = None
        else:
            macro_time_step_end = macro_time_step_end_val + self.macro_time_step_count_test
        return macro_time_step_end


sa.event.listens_for(Dataset, 'before_insert', propagate=True)(
    hydra_orm.utils.set_attr_to_func_value(Dataset, Dataset._processed_filename.key, hydra_orm.utils.generate_random_string)
)


class KuramotoSivashinsky1D(Dataset):
    domain_width: float = orm.make_field(orm.ColumnRequired(sa.Double), default=22.)
    domain_width_is_multiple_of_pi: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=64)
    coarse_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=16)

    @property
    def solution_dimension(self):
        return 1

    def dimensions(self):
        return [self.dimension]

    def coarse_dimensions(self):
        return [self.coarse_dimension]

    @property
    def embedding_dimension(self):
        return self.coarse_dimension

    @property
    def upsample_mode(self):
        return 'linear'


class BackwardFacingStep2D(Dataset):
    dimension_width: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    dimension_length: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    coarse_dimension_width: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=32)
    coarse_dimension_length: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=32)

    @property
    def solution_dimension(self):
        return 2

    def dimensions(self):
        return [self.dimension_width, self.dimension_length]

    def coarse_dimensions(self):
        return [self.coarse_dimension_width, self.coarse_dimension_length]

    @property
    def embedding_dimension(self):
        return self.coarse_dimension_width * self.coarse_dimension_length

    @property
    def upsample_mode(self):
        return 'bilinear'


class ChannelFlow3D(Dataset):
    dimension_width: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    dimension_length: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    dimension_height: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    coarse_dimension_width: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=8)
    coarse_dimension_length: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=32)
    coarse_dimension_height: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=8)

    @property
    def solution_dimension(self):
        return 3

    def dimensions(self):
        return [self.dimension_width, self.dimension_length, self.dimension_height]

    def coarse_dimensions(self):
        return [self.coarse_dimension_width, self.coarse_dimension_length, self.coarse_dimension_height]

    @property
    def embedding_dimension(self):
        return self.coarse_dimension_width * self.coarse_dimension_length * self.coarse_dimension_height

    @property
    def upsample_mode(self):
        return 'trilinear'
