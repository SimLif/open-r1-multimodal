#!/bin/bash

export NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eno2
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1

GPUS="0,1"
export CUDA_VISIBLE_DEVICES=$GPUS

# 取 worker0 第一个 port
# ports=($(echo $METIS_WORKER_0_PORT | tr ',' ' '))
# port=${ports[0]}
# port_in_cmd="$(echo "${METIS_WORKER_0_PORT:-2000}" | awk -F',' '{print $1}')"

# echo "total workers: ${ARNOLD_WORKER_NUM}"
# echo "cur worker id: ${ARNOLD_ID}"
# echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
# echo "master ip: ${METIS_WORKER_0_HOST}"
# echo "master port: ${port}"
# echo "master port in cmd: ${port_in_cmd}"

# export WANDB_BASE_URL=https://api.wandb.ai
# export WANDB_API_KEY="<PLACEHOLDER_WANDB_KEY_1>"
# wandb login $WANDB_API_KEY

# export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=med-moe-r1
# export WANDB_API_KEY="<PLACEHOLDER_WANDB_KEY_2>"
# export WANDB_RUN_NAME=Qwen-VL-2B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export WANDB_RUN_NAME=qwen2-vl-2b-moe-grpo-train_v2-ada-test
# wandb login $WANDB_API_KEY

cd ~/workspace/09-med-moe-r1
# pip3 install vllm==0.6.6.post1
# pip3 install -e ".[dev]"
# pip3 install wandb==0.18.3

# MODEL_PATH="/mnt/data/haoqiang/workspace/models/qwen2-vl-2b-instruct"
# MODEL_PATH="/mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-4e2-med-ada-5epoch"
# MODEL_PATH="/mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-4e-ada-1epoch"
MODEL_PATH="/mnt/data/haoqiang/workspace/05-moe-llava/checkpoints/qwen2-vl-2b-instruct-4e2-r1-1epoch"

torchrun --nproc_per_node=2 \
    --master_port=29500 \
    src/open_r1/grpo_mcq.py \
    --deepspeed ./local_scripts/zero2.json \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path $MODEL_PATH \
    --torch_dtype bfloat16 \
    --model_init_kwargs '{"torch_dtype": "bfloat16"}' \
    --dataset_name /mnt/data/haoqiang/workspace/data/medvqa-r1-pmc-vqa \
    --dataset_train_split train_v2 \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels $((324 * 28 * 28)) \
    --save_total_limit 8 \
    --num_train_epochs 1 \
    --run_name $WANDB_RUN_NAME \
    --optim paged_adamw_8bit