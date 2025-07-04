import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from einops import reduce

from imagen_pytorch import ElucidatedImagen, ImagenTrainer, Unet3D

from conf import conf, model
from g_led import utils
from g_led.transformer.sequentialModel import SequentialModel as Transformer


class Downsampler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.downsampler = nn.Upsample(size=cfg.coarse_dimensions(), mode=cfg.upsample_mode)

    def forward(self, batch, flatten_solution=False):
        batch_size, time_count = batch.shape[:2]
        batch_macro = self.downsampler(
            batch.view(-1, self.cfg.solution_dimension, *self.cfg.dimensions())
        ).view(batch_size, time_count, self.cfg.solution_dimension, *self.cfg.coarse_dimensions())
        if flatten_solution:
            batch_macro = batch_macro.view(batch_size, time_count, self.cfg.solution_dimension * self.cfg.embedding_dimension)
        return batch_macro


class Upsampler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.upsampler = nn.Upsample(size=cfg.dimensions(), mode=cfg.upsample_mode)

    def forward(self, batch):
        batch_size, time_count = batch.shape[:2]
        batch_micro = self.upsampler(
            batch.view(-1, self.cfg.solution_dimension, *self.cfg.coarse_dimensions())
        ).view(batch_size, time_count, self.cfg.solution_dimension, *self.cfg.dimensions())
        return batch_micro


class TrainSequential(pl.LightningModule):
    def __init__(self, cfg, downsampler, model):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.downsampler = downsampler
        self.model = model
        self.forecast_time_step_count = 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.get_model().learning_rate)
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.cfg.get_model().learning_rate_decay)
            ),
        )

    def setup(self, stage):
        pass

    def forecast(self, forecast_time_step_count, initial_sequence, flatten_solution=False):
        # process warm-up sequence of length 1 or more
        if (time_step_count := initial_sequence.shape[1]) > self.cfg.get_model().time_step_window_size:
            raise ValueError(
                f'The time step count of the initial sequence ({time_step_count}) must be less than or equal to model.time_step_window_size ({self.cfg.get_model().time_step_window_size}).'
                f' Please pass a initial sequence with at most {self.cfg.get_model().time_step_window_size} time steps, or set model.time_step_window_size={time_step_count} or larger.'
            )
        window_shifted_by_1_pred, cached_keys_values, *_ = self.model(inputs_embeds=initial_sequence, past=None)
        window_pred_batch = [
            initial_sequence[:, :1],  # save the initial state
            window_shifted_by_1_pred,
        ]
        window_pred = window_shifted_by_1_pred[:, -1:]  # iterate the latest state
        for time_step in range(1, forecast_time_step_count):  # start at 1 because we already forecasted one time step past the warm-up sequence
            if cached_keys_values[0][0].shape[2] < self.cfg.get_model().time_step_window_size:
                # cached_keys_values[*][0].shape[2] is the number of time steps processed in the trajectory (i.e., in the context of LLMs, the number of tokens in the context)
                window_shifted_by_1_pred, cached_keys_values, *_ = self.model(inputs_embeds=window_pred, past=cached_keys_values)
            else:
                # drop oldest key/value to maintain fixed context window
                cached_keys_values = [
                    [
                        # keys
                        cached_keys_values[layer][0][:, :, 1:],
                        # values
                        cached_keys_values[layer][1][:, :, 1:]
                    ]
                    for layer in range(self.cfg.get_model().attention_layer_count)
                ]
                window_shifted_by_1_pred, cached_keys_values, *_ = self.model(inputs_embeds=window_pred, past=cached_keys_values)
            window_pred = window_shifted_by_1_pred
            window_pred_batch.append(window_pred)
        window_pred_batch = torch.cat(window_pred_batch, dim=1)
        batch_size, time_count = window_pred_batch.shape[:2]

        window_pred_batch = window_pred_batch.view(batch_size, time_count, self.cfg.dataset.solution_dimension, *self.cfg.dataset.coarse_dimensions())
        if flatten_solution:
            window_pred_batch = window_pred_batch.view(batch_size, time_count, self.cfg.dataset.solution_dimension * self.cfg.dataset.embedding_dimension)

        return window_pred_batch

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        coarse_batch = self.downsampler(batch, flatten_solution=True)
        window = coarse_batch[:, :self.cfg.get_model().time_step_window_size, :]
        window_shifted_by_1_pred, *_ = self.model(inputs_embeds=window, past=None)
        window_shifted_by_1 = coarse_batch[:, 1:self.cfg.get_model().time_step_window_size+1, :]

        loss = F.mse_loss(window_shifted_by_1_pred, window_shifted_by_1)

        self.manual_backward(loss)
        optimizer.step()

        return dict(loss=loss)

    def validation_step(self, batch, _):
        batch, batch_idx, dataset_idx = batch
        coarse_batch = self.downsampler(batch, flatten_solution=True)
        initial_condition = coarse_batch[:, :self.cfg.get_model().initial_sequence_time_step_count]
        coarse_batch = coarse_batch[:, self.cfg.get_model().initial_sequence_time_step_count:self.cfg.get_model().initial_sequence_time_step_count+self.forecast_time_step_count]
        window_pred_batch = self.forecast(self.forecast_time_step_count, initial_condition, flatten_solution=True)[:, self.cfg.get_model().initial_sequence_time_step_count:]

        # local_batch_size = windows_pred.shape[0]
        relative_rmse_batch = reduce(
            (window_pred_batch - coarse_batch).square(),
            'batch time_step dim -> batch time_step',
            'mean'
        ).sqrt().mean(1) / reduce(coarse_batch.square(), 'batch time_step dim -> batch time_step', 'mean').sqrt().mean(1)

        return dict(
            relative_rmse_max=relative_rmse_batch.max(),
            relative_rmse_min=relative_rmse_batch.min(),
            relative_rmse_mean=relative_rmse_batch.mean(),
            relative_rmse_std=relative_rmse_batch.std(correction=0),
            relative_rmse_sum=relative_rmse_batch.sum(),
        )

    def predict_step(self, batch, batch_idx):
        raise RuntimeError('Use datasets.Macro to evaluate the sequential model with a specified initial sequence.')
        batch, batch_idx, dataset_idx = batch
        coarse_batch = self.downsampler(batch, flatten_solution=True)
        initial_sequence_time_step_count = self.cfg.get_model().initial_sequence_time_step_count
        initial_condition = coarse_batch[:, :initial_sequence_time_step_count]
        window_pred_batch = self.forecast(self.cfg.dataset.pred_forecast_time_step_count, initial_condition)
        return {
            utils.dataset_idx_to_dataset_name[dataset_idx]: window_pred_batch
        }


class TrainUpsampler(pl.LightningModule):
    def __init__(self, cfg, downsampler, upsampler, imagen_trainer):
        super().__init__()
        self.automatic_optimization = False
        self.strict_loading = False

        self.cfg = cfg
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.imagen_trainer = imagen_trainer

    def configure_optimizers(self):
        return None

    def setup(self, stage):
        pass

    def upsample(self, batch_macro):
        if isinstance(self.cfg.model, conf.Trained):
            time_step_window_size = self.cfg.model.conf.model.time_step_window_size
        else:
            time_step_window_size = self.cfg.model.time_step_window_size
        batch_macro_interpolated_to_micro = self.upsampler(batch_macro)
        batch_micro = self.imagen_trainer.sample(
            video_frames=time_step_window_size,
            cond_images=batch_macro_interpolated_to_micro.transpose(1, 2)
        ).transpose(1, 2)
        return batch_micro

    def training_step(self, batch, batch_idx):
        self.optimizers().step()  # increment global step for logging and checkpointing

        batch_macro_interpolated_to_micro = self.upsampler(self.downsampler(batch))
        loss = self.imagen_trainer(
            batch.transpose(1, 2),
            cond_images=batch_macro_interpolated_to_micro.transpose(1, 2),
            unet_number=1,
            ignore_time=False
        )
        self.imagen_trainer.update(unet_number=1)

        return dict(loss=torch.tensor(loss))

    def validation_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        batch, batch_idx, dataset_idx = batch
        batch_micro = self.upsample(self.downsampler(batch))
        return batch_micro


def get_model(cfg, ckpt_path=None):
    if isinstance(cfg.model, model.Transformer):
        downsampler = Downsampler(cfg.dataset)
        transformer = Transformer(cfg.model, cfg.dataset.solution_dimension * cfg.dataset.embedding_dimension)
        if ckpt_path is None:
            return TrainSequential(cfg, downsampler, transformer)
        else:
            return TrainSequential.load_from_checkpoint(ckpt_path, cfg=cfg, downsampler=downsampler, model=transformer)
    elif isinstance(cfg.model, model.Imagen):
        unet1 = Unet3D(
            dim=cfg.dataset.coarse_dimensions()[0],  # diff_args.unet_dim,
            cond_images_channels=cfg.dataset.solution_dimension,
            memory_efficient=True,
            dim_mults=(1, 2, 4, 8),  # mid: mid channel
        )
        width = cfg.dataset.dimensions()[0]
        imagen = ElucidatedImagen(
            unets=(unet1),
            image_sizes=width,
            image_width=width,
            channels=cfg.dataset.solution_dimension,   # Han Gao add the input to this args explicity
            random_crop_sizes=None,
            num_sample_steps=20,  # diff_args.num_sample_steps, # original is 10
            cond_drop_prob=0.1,
            sigma_min=0.002,
            sigma_max=80,        # max noise level, double the max noise level for upsampler (80, 160)
            sigma_data=0.5,      # standard deviation of data distribution
            rho=7,               # controls the sampling schedule
            P_mean=-1.2,         # mean of log-normal distribution from which noise is drawn for training
            P_std=1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn=80,          # parameters for stochastic sampling - depends on dataset, Table 5 in paper
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
            condition_on_text=False,
            auto_normalize_img=False  # Han Gao make it false
        )
        downsampler = Downsampler(cfg.dataset)
        upsampler = Upsampler(cfg.dataset)
        imagen_trainer = ImagenTrainer(imagen, device=torch.device(cfg.device))
        if ckpt_path is None:
            return TrainUpsampler(cfg, downsampler, upsampler, imagen_trainer)
        else:
            return TrainUpsampler.load_from_checkpoint(ckpt_path, cfg=cfg, downsampler=downsampler, upsampler=upsampler, imagen_trainer=imagen_trainer)
    elif isinstance(cfg.model, conf.Trained):
        return get_model(cfg.model.conf, ckpt_path=cfg.model.conf.run_dir/cfg.model.ckpt_filename)
    else:
        raise ValueError(f'Unknown model: {cfg}')
