cfg=cfg/reproduce/LPQLD_in21k_[2k].yaml
source scripts/gpu_utils.sh

LPQLD_imagenet_21k_result(){
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
    repeat=${11:-"0"}

    for model_size in 18; do
        model=resnet${model_size}

        # pre-defined paths for later generated labels
        # for path if ratio is 0.975, we need to make the path to be _975, otherwise cannot read
        ratio_path=$(echo $ratio | sed 's/0\.//g')
        # MS-50 meaning topk=50
        topk=$(echo $QUANT_METHOD | sed 's/.*-\([0-9]*\)/\1/g')
        fkd_path="syn_label_LPQLD/FKD_cutout_fp16_LPQLD_in21k_rn18_2K_ipc${ipc}_ratio${ratio_path}_topk${topk}"
        # fixed to rn18 due to we only use rn18 for recovering
        train_dir="/path/to/syn_data_in21k/sre2l_in21k_rn18_2K_ipc${ipc}"

        # gpus=$(find_best_gpu | tail -n 1)

        # check if the label is already generated
        if [ ! -d $fkd_path ]; then
            python generate_soft_label_pruning_batch.py --cfg_yaml $cfg --fkd_path $fkd_path --prune_ratio $ratio \
                --train_dir $train_dir --gpus $gpus --label_quantization ${QUANT_METHOD} 
        fi

        if [ "$repeat" != "0" ]; then
            run_name=LPQLD_in21k_ipc${ipc}_[${QUANT_METHOD}]_re${repeat}
        else
            run_name=LPQLD_in21k_ipc${ipc}_[${QUANT_METHOD}]
        fi

        # temperature scheduler command arguments
        temp_args=""
        if [ "$temp_scheduler" != "none" ]; then
            temp_args="--temp_scheduler $temp_scheduler"
            
            if [ "$temp_scheduler" == "cosine" ]; then
                temp_args="$temp_args --temp_min $temp_min"
            elif [ "$temp_scheduler" == "step" ]; then
                temp_args="$temp_args --temp_step_size $temp_step_size --temp_step_gamma $temp_step_gamma"
            fi

            # student temperature command arguments
            if [ "$temp_stu" != "-1.0" ]; then
                temp_args="$temp_args --temp_stu_dynamic $temp_stu"
            fi
        fi

        # gpus=$(find_best_gpu | tail -n 1)

        env WANDB_RUN_GROUP="LPQLD_in21k" \
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
            ${temp_args} &
        
        sleep 10
        
    done
}

for REPEAT in 1 2; do
    LPQLD_imagenet_21k_result 10 0.95 "MR-1000" 20 "step" 2 30 0.7 "1.0" 0 $REPEAT
    LPQLD_imagenet_21k_result 10 0.975 "MR-2000" 20 "step" 2 30 0.7 "1.0" 1 $REPEAT

    LPQLD_imagenet_21k_result 10 0.95 "MR-200" 20 "step" 2 30 0.7 "1.0" 2 $REPEAT
    LPQLD_imagenet_21k_result 10 0.975 "MR-400" 20 "step" 2 30 0.7 "1.0" 3 $REPEAT

    LPQLD_imagenet_21k_result 10 0.95 "MR-100" 20 "step" 2 30 0.7 "1.0" 3 $REPEAT
    LPQLD_imagenet_21k_result 10 0.975 "MR-200" 20 "step" 2 30 0.7 "1.0" 0 $REPEAT
done

# LPQLD_imagenet_21k_result 20 0.975 "MR-400" 20 "step" 2 30 0.7 "1.0" 3 0
# LPQLD_imagenet_21k_result 20 0.975 "MR-400" 20 "step" 2 30 0.7 "1.0" 3 0