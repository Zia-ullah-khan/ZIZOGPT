#!/bin/bash
# ZIZOGPT RL (DPO) Standalone Script
# This script resumes the pipeline from Step 4: Reinforcement Learning (DPO)

set -e

# Print header
cat <<EOM
======================================
ZIZOGPT RL (DPO) Standalone Script
======================================
Step 4/4: Reinforcement Learning (DPO)...
======================================
EOM

# Activate virtual environment if needed (uncomment and edit if required)
# source /path/to/venv/bin/activate

# Run RL (DPO) training
python scripts/run_rl.py \
    --model_name_or_path ./outputs/sft/final \
    --tokenizer_path ./tokenizer \
    --output_dir ./outputs/rl \
    --dataset_name nvidia/Nemotron-3-Nano-RL-Training-Blend \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --num_train_epochs 1 \
    --max_seq_length 2048 \
    --max_prompt_length 1024 \
    --max_response_length 1024 \
    --rl_algorithm dpo \
    --beta 0.1 \
    --loss_type sigmoid \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 100 \
    --save_total_limit 3 \
    --run_name zizogpt-rl \
    --report_to wandb \
    --overwrite_output_dir True \
    --seed 42

# Print completion message
echo "RL (DPO) training complete!"
