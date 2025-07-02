from dataclasses import field
from pathlib import Path
from typing import Any, List

import hydra_orm.utils
import omegaconf
import sqlalchemy as sa
from hydra_orm import orm

import conf.dataset
import conf.model
from g_led import utils


def get_engine(dir=str(utils.DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


class Conf(orm.Table):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(dataset=omegaconf.MISSING),
        dict(model=omegaconf.MISSING),
        '_self_',
    ])
    root_dir: str = field(default=str(utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((utils.DIR_ROOT/'..'/'..'/'out'/'g_led').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='output')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2376999025)
    fit: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    predict: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    dataset = orm.OneToManyField(conf.dataset.Dataset, required=True, default=omegaconf.MISSING)
    model = orm.OneToManyField(conf.model.Model, required=True, default=omegaconf.MISSING)

    def __post_init__(self):
        if isinstance(self.model, conf.model.Transformer) and self.dataset.time_step_window_size_train is not None and self.model.time_step_window_size >= self.dataset.time_step_window_size_train:
            raise ValueError(
                'model.time_step_window_size must be at least one less than the dataset.time_step_window_size_train so that the model can attempt to predict at least one time step,'
                f' but model.time_step_window_size={self.model.time_step_window_size} >= dataset.time_step_window_size_train={self.dataset.time_step_window_size_train}.'
            )

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id

    def get_model(self):
        if isinstance(self.model, Trained):
            return self.model.conf.model
        else:
            return self.model


sa.event.listens_for(Conf, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


class Trained(conf.model.Model):
    conf = orm.OneToManyField(Conf, default=omegaconf.MISSING, enforce_element_type=False)
    ckpt_filename: str = orm.make_field(orm.ColumnRequired(sa.String(len('epoch_####.ckpt'))), default='last.ckpt')

    @staticmethod
    def transform_conf(session, conf_alt_id):
        if conf_alt_id == omegaconf.MISSING:
            raise ValueError('Please set a conf alt_id with model.conf=<conf_alt_id>.')
        conf = (
            sa.select(Conf)
            .where(Conf.alt_id == conf_alt_id)
        )
        conf = session.execute(conf)
        conf = list(zip(range(2), conf))
        assert len(conf) == 1
        conf = conf[0][1][0]
        return conf


orm.store_config(Conf)
orm.store_config(conf.dataset.KuramotoSivashinsky1D, group=Conf.dataset.key, name=f'_{conf.dataset.KuramotoSivashinsky1D.__name__}')
orm.store_config(conf.dataset.BackwardFacingStep2D, group=Conf.dataset.key, name=f'_{conf.dataset.BackwardFacingStep2D.__name__}')
orm.store_config(conf.model.Transformer, group=Conf.model.key, name=f'_{conf.model.Transformer.__name__}')
orm.store_config(conf.model.Imagen, group=Conf.model.key, name=f'_{conf.model.Imagen.__name__}')
orm.store_config(Trained, group=Conf.model.key)
