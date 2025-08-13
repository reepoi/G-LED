#!/bin/bash
#COBALT -t 0
#COBALT -n 1
#COBALT --cwd /home/ttransue/GitHub/G-LED
source .venv/bin/activate
python src/g_led/main_sequential.py +experiment=TrainTransformerBackwardFacingStep2DOverfit dataset/downsampler=DownsamplerGaussian dataset/upsampler=UpsamplerLinear
