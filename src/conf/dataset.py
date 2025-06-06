from typing import List, Any

import hydra_orm.utils
from hydra_orm import orm


class Dataset(orm.InheritableTable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        '_self_',
    ])
