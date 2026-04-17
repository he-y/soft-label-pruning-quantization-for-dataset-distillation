cfg=cfg/reproduce/LPQLD_in1k_[4k].yaml
source scripts/gpu_utils.sh

LPQLD_imagenet_1k_result(){
    ipc=$1
    ratio=$2
    QUANT_METHOD=${3:-"None"}

    temperature=$4
    temp_scheduler=$5
    temp_min=$6
    temp_step_size=$7
    temp_step_gamma=$8
    temp_stu=${9:-"-1.0"}

    gpus=${10:-"0"}

    for model_size in 18; do
        model=resnet${model_size}

        # pre-defined paths for later generated labels
        # for path if ratio is 0.975, we need to make the path to be _975, otherwise cannot read
        ratio_path=$(echo $ratio | sed 's/0\.//g')
        # MS-50 meaning topk=50
        topk=$(echo $QUANT_METHOD | sed 's/.*-\([0-9]*\)/\1/g')
        fkd_path="syn_label_LPQLD/FKD_cutmix_fp16_LPQLD_in1k_rn18_4k_ipc${ipc}_ratio${ratio_path}_topk${topk}"
        # fixed to rn18 due to we only use rn18 for recovering
        train_dir="/path/to/syn_data_LPLD/LPLD_in1k_rn18_4k_ipc${ipc}"

        # gpus=$(find_best_gpu | tail -n 1)

        # check if the label is already generated
        if [ ! -d $fkd_path ]; then
            python generate_soft_label_pruning_batch.py --cfg_yaml $cfg --fkd_path $fkd_path --prune_ratio $ratio \
                --train_dir $train_dir --gpus $gpus --label_quantization ${QUANT_METHOD} 
        fi

        run_name=LPQLD_rn18_ipc${ipc}_[${QUANT_METHOD}]

        # temperature scheduler command arguments
        temp_args=""
        if [ "$temp_scheduler" != "none" ]; then
            temp_args="--temp_scheduler $temp_scheduler"
            
            if [ "$temp_scheduler" == "cosine" ]; then
                temp_args="$temp_args --temp_min $temp_min"
            elif [ "$temp_scheduler" == "step" ]; then
                temp_args="$temp_args --temp_step_size $temp_step_size --temp_step_gamma $temp_step_gamma"
            fi
            if [ "$temp_stu" != "-1.0" ]; then
                temp_args="$temp_args --temp_stu_dynamic $temp_stu"
            fi
        fi

        env WANDB_RUN_GROUP="table_1" \
        python train_FKD_LPQLD.py \
            --model ${model} \
            --prune_ratio ${ratio} \
            --sample_metric 'random' \
            --gpus $gpus \
            --cfg_yaml $cfg \
            --fkd_path ${fkd_path} \
            --train_dir ${train_dir} \
            --run_name ${run_name} \
            --label_quantization ${QUANT_METHOD} \
            --exp_name LPQLD \
            --temperature ${temperature} \
            --output_dir ./save/LPQLD_${QUANT_METHOD} \
            ${temp_args} &
        
        sleep 10
        
    done
}

## example usage:
# LPQLD_imagenet_1k_result [IPC]
# 0 -> 1x
# 0.9 -> 10x
# 0.95 -> 20x
# 0.97 -> 30x
# 0.975 -> 40x
for IPC in 10 20 50 100 200; do
    # usage:
    # LPQLD_imagenet_1k_result [IPC] [RATIO] [QUANT_METHOD] [TEMPERATURE] [TEMP_SCHEDULER] [TEMP_MIN] [TEMP_STEP_SIZE] [TEMP_STEP_GAMMA] [TEMP_STU]
    LPQLD_imagenet_1k_result $IPC 0.9 "MR-10" 20 "step" 2 30 0.7 "1.0" 0
    LPQLD_imagenet_1k_result $IPC 0.9 "MR-50" 20 "step" 2 30 0.7 "1.0" 1
    LPQLD_imagenet_1k_result $IPC 0.9 "MR-100" 20 "step" 2 30 0.7 "1.0" 2
    LPQLD_imagenet_1k_result $IPC 0.9 "MR-250" 20 "step" 2 30 0.7 "1.0" 3
    wait
done