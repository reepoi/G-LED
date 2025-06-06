from dataclasses import field
from pathlib import Path
from typing import Any, List

import hydra_orm.utils
import omegaconf
import sqlalchemy as sa
from hydra_orm import orm

import conf.dataset
from g_led import utils


def get_engine(dir=str(utils.DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


class Conf(orm.Table):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(dataset=omegaconf.MISSING),
        # dict(model=omegaconf.MISSING),
        '_self_',
    ])
    root_dir: str = field(default=str(utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((utils.DIR_ROOT/'..'/'..'/'out'/'g_led').resolve()))
    _data_dir: str = field(default=str(Path('/mnta/taosData/diffusion-dynamics/G-LED/data').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='output')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2376999025)
    fit: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    dataset = orm.OneToManyField(conf.dataset.Dataset, required=True, default=omegaconf.MISSING)
    # model = orm.OneToManyField(models.Model, required=True, default=omegaconf.MISSING)

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id

    @property
    def data_dir(self):
        return Path(self._data_dir)


sa.event.listens_for(Conf, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


orm.store_config(Conf)
orm.store_config(conf.dataset.Dataset, group=Conf.dataset.key)
