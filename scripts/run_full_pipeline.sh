#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# ==============================================================================
# ZIZOGPT FULL TRAINING PIPELINE (4x B200)
# ==============================================================================
# Sequence:
# 1. Pre-training (Scratch 1.5B Model on FineWeb-Edu 10B Tokens)
# 2. SFT (Supervised Fine-Tuning on UltraChat)
# 3. DPO (Direct Preference Optimization on UltraFeedback)
# ==============================================================================

# --- Environment Setup ---
export TORCH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1  # Faster downloads

echo "======================================================================"
echo "   STARTING ZIZOGPT PIPELINE"
echo "======================================================================"

# --- Step 1: Pre-training ---
echo ""
echo ">>> [Stage 1/3] Pre-training (Scratch Build)..."
echo "Dataset: HuggingFaceFW/fineweb-edu (sample-10BT)"
echo "Config: configs/pretrain_config_flat.yaml"

deepspeed --num_gpus=4 scripts/run_pretrain.py \
    configs/pretrain_config_flat.yaml

if [ $? -ne 0 ]; then
    echo "!!! Pre-training FAILED. Aborting pipeline."
    exit 1
fi

echo ">>> Pre-training COMPLETE."

# --- Step 2: Supervised Fine-Tuning (SFT) ---
echo ""
echo ">>> [Stage 2/3] Supervised Fine-Tuning (SFT)..."
echo "Dataset: HuggingFaceH4/ultrachat_200k"
echo "Base Model: outputs/pretrain/final"

# Explicitly set the dataset name in the command to override defaults
deepspeed --num_gpus=4 scripts/run_sft.py \
    configs/sft_config_flat.yaml \
    --model_name_or_path "./outputs/pretrain/final" \
    --dataset_name "HuggingFaceH4/ultrachat_200k"

if [ $? -ne 0 ]; then
    echo "!!! SFT FAILED. Aborting pipeline."
    exit 1
fi

echo ">>> SFT COMPLETE."

# --- Step 3: Direct Preference Optimization (DPO) ---
echo ""
echo ">>> [Stage 3/3] Direct Preference Optimization (DPO)..."
echo "Dataset: HuggingFaceH4/ultrafeedback_binarized"
echo "Base Model: outputs/sft/merged"

# Check if merged model exists (SFT script should merge LoRA)
if [ ! -d "./outputs/sft/merged" ]; then
    echo "WARNING: Merged SFT model not found at ./outputs/sft/merged"
    echo "Checking for final checkpoint..."
    if [ -d "./outputs/sft/final" ]; then
        echo "Using ./outputs/sft/final (Adapters will need to be loaded)"
        MODEL_PATH="./outputs/sft/final"
        # Note: DPO script handles adapter loading if detected
    else
        echo "!!! No SFT model found. Aborting."
        exit 1
    fi
else
    MODEL_PATH="./outputs/sft/merged"
fi

deepspeed --num_gpus=4 scripts/run_rl.py \
    configs/dpo_config_flat.yaml \
    --model_name_or_path "$MODEL_PATH" \
    --ref_model_name_or_path "$MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "!!! DPO FAILED. Aborting pipeline."
    exit 1
fi

echo "======================================================================"
echo "   ZIZOGPT PIPELINE COMPLETE"
echo "   Final Model: ./outputs/rl/final"
echo "======================================================================"