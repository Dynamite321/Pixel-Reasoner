
export WANDB_PROJECT=pixel_reasoner_openrlhf

export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

RUN_NAME=sft
export LOG_PATH="./debug_log_$RUN_NAME.log"

python -m torch.distributed.run --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    instruction_tuning/sft_tool.py \
    --deepspeed instruction_tuning/local_scripts/zero2.json \
    --output_dir /fsx-project/yanjunfu/checkpoints/Pixel-Reasoner/sft \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct\
    --datasetpath  /fsx-project/yanjunfu/datasets/Pixel-Reasoner/sft/train.json\
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --eval_strategy no \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --bf16 \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --data_seed 49 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 5 \
    --run_name $RUN_NAME \
    --save_strategy epoch \
    --save_steps 100 \
    --save_only_model true \
    --freeze_vision_modules true \
    2>&1 | tee $LOG_PATH


    