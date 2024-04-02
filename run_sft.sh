#!/bin/bash

BASE_DIR=${BASE_DIR:-/data/home/zfyang/home/oishi}

MODEL=${BASE_DIR}${MODEL:-/model/llemma-7b}
CONFIG=${BASE_DIR}${CONFIG:-/llemma-ft/ds_config.json}
OUTDIR=${BASE_DIR}${OUTDIR:-/model/llemma-7b-v1}
DATADIR=${BASE_DIR}${DATADIR:-/llemma-ft/data/random/train_alpaca.json}


deepspeed --include localhost:0 ${BASE_DIR}/llemma-ft/sft.py \
    --deepspeed $CONFIG \
    --model_name_or_path $MODEL \
    --dataset_path $DATADIR \
    --use_lora \
    --lora_target q_proj,v_proj \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --output_dir $OUTDIR \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --save_steps 100 \
    --logging_steps 10 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --warmup_steps 20 \
    --num_train_epochs 3.0 \
    --bf16
    