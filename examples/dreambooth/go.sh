#!/bin/bash

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="1girl" \
  --prompt_file_path="/shared/captions2.yml" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam
  --gradient_checkpointing \
  --learning_rate=1e-7 \
  --lr_scheduler="onecycle" \
  --lr_warmup_steps=0 \
  --max_train_steps=800
