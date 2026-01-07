#!/bin/bash
set -e

echo "Cleaning up HF stack..."
pip uninstall -y transformers accelerate tokenizers trl peft bitsandbytes

echo "Reinstalling HF stack (Aligned Versions)..."
# Using known stable versions for early 2025 context
pip install --ignore-installed \
    transformers>=4.48.0 \
    accelerate>=1.2.0 \
    tokenizers>=0.21.0 \
    trl>=0.13.0 \
    peft>=0.14.0 \
    bitsandbytes>=0.45.0 \
    huggingface-hub

echo "Verifying installation..."
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
