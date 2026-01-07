#!/bin/bash
set -e  # Exit on error

echo "=========================================="
echo "   ZIZOGPT Vast.ai Setup Script"
echo "=========================================="

# 1. System Dependencies
echo "[1/5] Installing System Dependencies..."
apt-get update && apt-get install -y git ninja-build libaio-dev python3-pip

# 2. Python Requirements
echo "[2/5] Installing Python Libraries..."
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch (assuming pre-installed on good images, but verifying)
# For Blackwell, we'd ideally want PyTorch Nightly or 2.5+, but standard 2.4 is safe default
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install Project Deps
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# 3. Authentication
echo "[3/5] Authentication Setup"
if [ -z "$HF_TOKEN" ]; then
    echo "Please log in to Hugging Face (Required for Llama 3.2 tokenizer and Nemotron datasets):"
    huggingface-cli login
else
    echo "HF_TOKEN found, logging in..."
    huggingface-cli login --token "$HF_TOKEN"
fi

echo "Setting up WandB (Optional, press Enter to skip if not needed):"
wandb login

# 4. Tokenizer Setup
echo "[4/5] Training Custom Tokenizer (Vocab 32k)..."
python3 scripts/train_tokenizer.py --vocab_size 32768

# 5. Cluster Verification
echo "[5/5] Verifying Cluster Status..."
python3 verify_cluster.py

echo "=========================================="
echo "   Setup Complete! Ready to Train."
echo "=========================================="
echo "Run the following command to start training:"
echo "accelerate launch --config_file configs/accelerate_config.yaml scripts/run_pretrain.py configs/pretrain_config.yaml"
