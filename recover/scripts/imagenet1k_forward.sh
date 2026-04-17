#!/bin/bash

cd imagenet1k_forward

# make the output directory
OUTPUT_DIR=../model_with_class_bn
mkdir -p $OUTPUT_DIR

DATA_PATH="/path/to/imagenet"  # Set this to your ImageNet dataset path

torchrun --nproc_per_node=4 train.py --model resnet18 \
    --data-path $DATA_PATH \
    --batch-size 64 --output-dir $OUTPUT_DIR >> $OUTPUT_DIR/log_in1k.txt

cd ..