cd tiny_forward

# make the output directory
OUTPUT_DIR=../model_with_class_bn
mkdir -p $OUTPUT_DIR

DATA_PATH="/path/to/tiny-imagenet-200"  # Set this to your Tiny-ImageNet dataset path

# check if resnet18_tiny_ep50.pth exists
if [ ! -f "resnet18_tiny_ep50.pth" ]; then
    echo "Downloading resnet18_tiny_ep50.pth from \"zeyuanyin/tiny-imagenet\""
    wget https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn18_50ep/checkpoint.pth -O resnet18_tiny_ep50.pth
fi

wandb disabled
torchrun --nproc_per_node=1 forward_once.py \
    --model 'resnet18' \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir $OUTPUT_DIR \
    --data-path $DATA_PATH

cd ..