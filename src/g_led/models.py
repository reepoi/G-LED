import torch

from imagen_pytorch import ElucidatedImagen, ImagenTrainer, Unet3D

from conf import conf, model
from g_led.transformer.sequentialModel import SequentialModel as Transformer


def get_model(cfg):
    if isinstance(cfg.model, model.Transformer):
        ckpt_path = None
        return Transformer(cfg.model, cfg.dataset.solution_dimension * cfg.dataset.embedding_dimension), ckpt_path
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
        ckpt_path = None
        return ImagenTrainer(imagen, device=torch.device(cfg.device)), ckpt_path
    elif isinstance(cfg.model, conf.Trained):
        ckpt_path = cfg.model.conf.run_dir/cfg.model.ckpt_filename
        m, _ = get_model(cfg.model.conf)
        return m, ckpt_path
    else:
        raise ValueError(f'Unknown model: {cfg}')
