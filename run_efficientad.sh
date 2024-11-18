#!/bin/bash

python efficientad.py \
    --dataset mvtec_ad \
    --subdataset bottle \
    --output_dir output/1 \
    --model_size small \
    --weights models/teacher_small.pth \
    --imagenet_train_path none \
    --mvtec_ad_path ./mvtec_anomaly_detection \
    --mvtec_loco_path ./mvtec_loco_anomaly_detection \
    --train_steps 70000
