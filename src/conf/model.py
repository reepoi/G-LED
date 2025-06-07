from dataclasses import field
from hydra_orm import orm
import sqlalchemy as sa


class Model(orm.InheritableTable):
    pass


class Trainable(Model):
    epoch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    learning_rate: float =  orm.make_field(orm.ColumnRequired(sa.Double), default=0)
    learning_rate_decay: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0)


class Transformer(Trainable):
    embd_pdrop: float = field(default=0.)
    attn_pdrop: float = field(default=0.)
    resid_pdrop: float = field(default=0.)

    attention_layer_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    attention_head_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    time_step_window_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    layer_norm_epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-5)
    weight_init_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.02)
    output_hidden_states: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    output_attentions: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    march_tolerance: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.01)
