#!/bin/bash
#COBALT -t 60
#COBALT -n 1
#COBALT --cwd /home/ttransue/GitHub/G-LED
source .venv/bin/activate
# python src/g_led/main_upsampler.py +experiment=EvalTrainedImagenBackwardFacingStep2D dataset/downsampler@dataset.dataset.downsampler=DownsamplerGaussian dataset/upsampler@dataset.dataset.upsampler=UpsamplerLinear dataset.split=TRAIN dataset.initial_sequence_time_step_start=0 dataset.initial_sequence_time_step_count=1 model.conf=y1nxregp model.ckpt_filename=epoch_179.ckpt
python src/g_led/main_upsampler.py +experiment=EvalTrainedImagenBackwardFacingStep2D dataset/downsampler@dataset.dataset.downsampler=DownsamplerGaussian dataset/upsampler@dataset.dataset.upsampler=UpsamplerLinear dataset.split=TRAIN dataset.initial_sequence_time_step_start=0 dataset.initial_sequence_time_step_count=1 model.conf=dmkj71or model.ckpt_filename=epoch_1199.ckpt
