#!/bin/bash
set -e

echo "Cleaning up HF stack..."
pip uninstall -y transformers accelerate tokenizers trl peft bitsandbytes

echo "Reinstalling HF stack (Exact Stable Versions)..."
# Pin exact versions to avoid resolution backtracking
pip install --ignore-installed \
    transformers==4.46.0 \
    accelerate==1.0.1 \
    tokenizers==0.20.1 \
    trl==0.11.4 \
    peft==0.13.2 \
    bitsandbytes==0.44.1 \
    huggingface-hub==0.26.2

echo "Verifying installation..."
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
