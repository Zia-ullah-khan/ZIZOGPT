#!/bin/bash
# Full training pipeline for ZIZOGPT (from scratch)
# This script runs tokenizer training, pre-training, SFT, and RL in sequence

set -e  # Exit on error

# Configuration
export WANDB_PROJECT="zizogpt-from-scratch"
export HF_HOME="./cache/huggingface"
export TOKENIZERS_PARALLELISM="false"

# GPU settings (optimized for a single H100)
export CUDA_VISIBLE_DEVICES="0"
TOKENIZER_PATH="./tokenizer"

echo "======================================"
echo "ZIZOGPT Full Training Pipeline (From Scratch)"
echo "======================================"

# 1. Train Tokenizer
echo ""
echo "Step 1/4: Training Tokenizer..."
echo "======================================"

python scripts/train_tokenizer.py \
    --vocab_size 128000 \
    --output_dir $TOKENIZER_PATH

echo "Tokenizer training complete!"

# 2. Pre-training
echo ""
echo "Step 2/4: Pre-training..."
echo "======================================"

python scripts/run_pretrain.py \
    --from_scratch true \
    --architecture llama \
    --tokenizer_name $TOKENIZER_PATH \
    --hidden_size 2048 \
    --intermediate_size 5504 \
    --num_hidden_layers 24 \
    --num_attention_heads 16 \
    --num_key_value_heads 16 \
    --vocab_size 128000 \
    --max_position_embeddings 262144 \
    --use_sample_dataset true \
    --max_seq_length 4096 \
    --streaming false \
    --output_dir ./outputs/pretrain \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 3e-4 \
    --num_train_epochs 1 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to wandb \
    --run_name zizogpt-pretrain-scratch

echo "Pre-training complete!"

# 3. Supervised Fine-Tuning
echo ""
echo "Step 3/4: Supervised Fine-Tuning..."
echo "======================================"

python scripts/run_sft.py \
    --model_name_or_path ./outputs/pretrain/final \
    --tokenizer_name $TOKENIZER_PATH \
    --use_lora true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --max_seq_length 4096 \
    --dataset_subset chat \
    --output_dir ./outputs/sft \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to wandb \
    --run_name zizogpt-sft

echo "SFT complete!"

# 4. Reinforcement Learning (DPO)
echo ""
echo "Step 4/4: Reinforcement Learning (DPO)..."
echo "======================================"

python scripts/run_rl.py \
    --model_name_or_path ./outputs/sft/final \
    --tokenizer_name $TOKENIZER_PATH \
    --use_lora true \
    --lora_r 32 \
    --lora_alpha 64 \
    --rl_algorithm dpo \
    --beta 0.1 \
    --max_seq_length 2048 \
    --output_dir ./outputs/rl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --num_train_epochs 1 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 200 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to wandb \
    --run_name zizogpt-rl

echo ""
echo "======================================"
echo "Training Pipeline Complete!"
echo "======================================"
echo "Final model saved to: ./outputs/rl/final"
