#!/bin/bash

 /home/zf/Thesis/EfficientAD/venv/bin/python3.12 efficientad.py \
    --dataset mvtec_ad \
    --subdataset bottle \
    --output_dir output/1 \
    --model_size medium \
    --weights models/teacher_medium.pth \
    --imagenet_train_path none \
    --mvtec_ad_path  /home/zf/Thesis/Datasets/Real-IAD \
    --train_steps 70000
