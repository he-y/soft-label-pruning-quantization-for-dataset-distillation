: <<'COMMENT'
    This script provides utility functions for finding the best GPU to run a job on.
    It is intended to be sourced in a bash script that starts a training job.
    The script will continuously check for the best available GPU and start the job on that GPU.
    If no GPU is available, the script will wait for 60 seconds and try again.
    The script will exit once the job has been started.

    Usage:
    1. Source this script in your bash script.
    2. Use the `find_best_gpu` function to find the best GPU to run the job on.
    3. Start the training job with the selected GPU.

    Example:
    ```bash
    #!/bin/bash

    # Source the utility functions
    source gpu_utils.sh

    while true; do 
        CUDA_DEVICE=$(find_best_gpu | tail -n 1)
        echo "CUDA_DEVICE: $CUDA_DEVICE"
        if [[ $CUDA_DEVICE -ge 0 ]]; then
            echo "**Starting training job with GPU $CUDA_DEVICE**"

            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python <PYTHON_FILE>.py

            sleep 120  # Wait for 60 seconds before checking for the next job
            break  # Exit the loop once we've started the job
        else
            echo "No GPU available, waiting for 60 seconds..."
            sleep 60
        fi
    done
    ```
    To exlude certain GPUs from being selected, pass the GPU indices as a comma-separated string to the `find_best_gpu` function.
    Pass the required memory for the job as the second argument to the function.
    max(approximated_memory, most_recent_job_memory) will be used as the required memory for the job.
    Example:
    find_best_gpu "0,2"  # Exclude GPUs 0 and 2 from selection
    or
    find_best_gpu "0,2" 4096  # Exclude GPUs 0 and 2 from selection and require 4096 MiB of memory for the job
COMMENT

function find_best_gpu {
    # Accept GPUs to exclude as first argument and required memory as second argument
    excluded_gpus="$1"
    required_memory="${2:-0}"  # Default to 0 if not specified

    # Query GPU memory and running processes
    gpu_info=$(nvidia-smi --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader,nounits)
    gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
    current_user=$(whoami)

    echo "=== GPU Memory Information ==="
    echo "$gpu_info" | awk -F, '{printf "GPU %d: UUID:%s, Total Memory: %s MiB, Free Memory: %s MiB\n", $1, $2, $3, $4}'
    echo ""

    echo "=== GPU Process Information ==="
    echo "$gpu_processes" | awk -F, '{printf "GPU UUID:%s, PID:%s, Used Memory: %s MiB\n", $1, $2, $3}'
    echo ""

    # Find the most recent process of current user across all GPUs
    most_recent_pid=""
    most_recent_memory=0
    most_recent_gpu_uuid=""

    while IFS=',' read -r gpu_uuid pid memory; do
        gpu_uuid=$(echo "$gpu_uuid" | tr -d ' ')
        pid=$(echo "$pid" | tr -d ' ')
        memory=$(echo "$memory" | tr -d ' ')

        pid_user=$(ps -o user= -p "$pid" 2>/dev/null)
        if [[ "$pid_user" == "$current_user" ]]; then
            start_time=$(ps -o lstart= -p "$pid" 2>/dev/null | date -f - +%s 2>/dev/null)
            if [[ -n "$start_time" ]]; then
                if [[ -z "$most_recent_pid" ]] || [[ $start_time -gt $(ps -o lstart= -p "$most_recent_pid" 2>/dev/null | date -f - +%s 2>/dev/null) ]]; then
                    most_recent_pid=$pid
                    most_recent_memory=$memory
                    most_recent_gpu_uuid=$gpu_uuid
                fi
            fi
        fi
    done <<< "$gpu_processes"

    # Determine the memory requirement
    memory_requirement=$(( required_memory > most_recent_memory ? required_memory : most_recent_memory ))

    echo "Most recent job (User: $current_user):"
    echo "PID: $most_recent_pid"
    echo "Memory: $most_recent_memory MiB"
    echo "GPU UUID: $most_recent_gpu_uuid"
    echo "Required Memory: $required_memory MiB"
    echo "Memory Requirement for New Job: $memory_requirement MiB"
    echo ""

    best_gpu=-1
    max_available_memory=0

    echo "=== Analyzing GPUs ==="

    while IFS=',' read -r gpu_index gpu_uuid total_memory free_memory; do
        gpu_uuid=$(echo "$gpu_uuid" | tr -d ' ')
        
        # Skip the excluded GPUs
        if [[ ",$excluded_gpus," == *",$gpu_index,"* ]]; then
            echo "GPU $gpu_index is excluded."
            continue
        fi
        
        echo "GPU $gpu_index:"
        echo "  UUID: $gpu_uuid"
        echo "  Total Memory: $total_memory MiB"
        echo "  Free Memory: $free_memory MiB"
        echo "  Required Memory for New Job: $memory_requirement MiB"
        echo "  Remaining Memory: $((free_memory - memory_requirement)) MiB"

        # Check if free_memory is enough for the calculated memory_requirement
        if ((free_memory >= memory_requirement)); then
            if ((free_memory > max_available_memory)); then
                max_available_memory=$free_memory
                best_gpu=$gpu_index
                echo "  -> This GPU is currently the best candidate."
            fi
        fi
        echo ""
    done <<< "$gpu_info"

    if [[ $best_gpu -ge 0 ]]; then
        echo "Best GPU for the job: GPU $best_gpu (Remaining Memory: $max_available_memory MiB)"
    else
        echo "No suitable GPU found for the job."
    fi

    echo "$best_gpu" >&2  # Print all status messages to stderr
    printf "%d" "$best_gpu"  # Return only the number to stdout
}