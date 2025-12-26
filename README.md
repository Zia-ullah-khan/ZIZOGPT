# ZIZOGPT - LLM Training Pipeline

A comprehensive training pipeline for building your own Large Language Model using NVIDIA Nemotron datasets.

## 🚀 Features

- **Pre-training**: Train a language model from scratch using Nemotron pre-training datasets
- **Supervised Fine-Tuning (SFT)**: Fine-tune on instruction-following datasets
- **Reinforcement Learning (RL)**: Align model behavior using DPO/PPO
- **Efficient Training**: Support for LoRA, 4-bit quantization, DeepSpeed, and Flash Attention 2
- **Flexible Configuration**: YAML-based configuration for easy customization

## 📁 Project Structure

```
ZIZOGPT/
├── configs/
│   ├── pretrain_config.yaml    # Pre-training configuration
│   ├── sft_config.yaml         # SFT configuration
│   ├── rl_config.yaml          # RL training configuration
│   └── deepspeed_zero3.json    # DeepSpeed ZeRO-3 config
├── scripts/
│   ├── run_pretrain.py         # Pre-training entry point
│   ├── run_sft.py              # SFT entry point
│   ├── run_rl.py               # RL training entry point
│   ├── run_full_pipeline.sh    # Full pipeline (Linux/Mac)
│   └── run_full_pipeline.ps1   # Full pipeline (Windows)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Dataset loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_builder.py    # Model creation and configuration
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py         # Pre-training script
│   │   ├── sft.py              # SFT training script
│   │   └── rl_training.py      # RL training script
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Utility functions
├── outputs/                     # Training outputs (created during training)
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### 1. Clone the repository

```bash
cd E:\Projects\ZIZOGPT
```

### 2. Create a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Login to HuggingFace (required for accessing Nemotron datasets)

```bash
huggingface-cli login
```

### 5. (Optional) Login to Weights & Biases for experiment tracking

```bash
wandb login
```

## 📊 NVIDIA Nemotron Datasets

This pipeline uses NVIDIA's Nemotron datasets:

### Pre-training Datasets
- `nvidia/Nemotron-CC-v2.1` - 3.8B tokens of curated web data
- `nvidia/Nemotron-CC-v2` - 8.79B tokens of web data
- `nvidia/Nemotron-Pretraining-Code-v2` - 836M tokens of code
- `nvidia/Nemotron-CC-Math-v1` - 190M tokens of math content
- `nvidia/Nemotron-Pretraining-Specialized-v1` - 60.7M specialized tokens
- `nvidia/Nemotron-Pretraining-Dataset-sample` - Sample dataset for testing

### Post-training Datasets (SFT)
- `nvidia/Nemotron-Instruction-Following-Chat-v1` - 288k instruction samples
- `nvidia/Nemotron-Math-v2` - 1.95M math samples
- `nvidia/Nemotron-Math-Proofs-v1` - 925k math proof samples
- `nvidia/Nemotron-Science-v1` - 226k science samples
- `nvidia/Nemotron-Agentic-v1` - 978 agentic samples
- `nvidia/Nemotron-Competitive-Programming-v1` - 3.11k programming samples

### RL Training
- `nvidia/Nemotron-3-Nano-RL-Training-Blend` - RL training blend

## 🏃 Quick Start

### Option 1: Run the Full Pipeline

```powershell
# Windows
.\scripts\run_full_pipeline.ps1
```

```bash
# Linux/Mac
./scripts/run_full_pipeline.sh
```

### Option 2: Run Individual Steps

#### Pre-training (from scratch)

```bash
python scripts/run_pretrain.py \
    --from_scratch true \
    --architecture llama \
    --hidden_size 2048 \
    --num_hidden_layers 24 \
    --use_sample_dataset true \
    --output_dir ./outputs/pretrain \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --bf16 true
```

#### Pre-training (continue from checkpoint)

```bash
python scripts/run_pretrain.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_sample_dataset true \
    --output_dir ./outputs/pretrain \
    --per_device_train_batch_size 4 \
    --bf16 true
```

#### Supervised Fine-Tuning

```bash
python scripts/run_sft.py \
    --model_name_or_path ./outputs/pretrain/final \
    --use_lora true \
    --lora_r 64 \
    --dataset_subset chat \
    --output_dir ./outputs/sft \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 true
```

#### Reinforcement Learning (DPO)

```bash
python scripts/run_rl.py \
    --model_name_or_path ./outputs/sft/final \
    --rl_algorithm dpo \
    --beta 0.1 \
    --output_dir ./outputs/rl \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-7 \
    --bf16 true
```

## ⚙️ Configuration Options

### Model Architectures

- `llama` - LLaMA-style architecture (default)
- `mistral` - Mistral architecture
- `gpt2` - GPT-2 architecture

### Model Sizes (Examples)

| Size | Hidden | Layers | Heads | Params |
|------|--------|--------|-------|--------|
| Tiny | 512 | 8 | 8 | ~125M |
| Small | 1024 | 12 | 12 | ~350M |
| Medium | 2048 | 24 | 16 | ~1.3B |
| Large | 4096 | 32 | 32 | ~7B |
| XL | 5120 | 40 | 40 | ~13B |

### LoRA Configuration

```bash
--use_lora true \
--lora_r 64 \
--lora_alpha 128 \
--lora_dropout 0.05 \
--lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### Quantization (for memory efficiency)

```bash
--use_4bit true  # 4-bit quantization (QLoRA)
--use_8bit true  # 8-bit quantization
```

## 🖥️ Hardware Requirements

### Minimum (for testing with sample dataset)
- GPU: 1x NVIDIA GPU with 8GB VRAM
- RAM: 16GB
- Storage: 50GB

### Recommended (for full training)
- GPU: 8x NVIDIA A100 (40GB or 80GB)
- RAM: 256GB+
- Storage: 2TB+ NVMe SSD

### Memory-Efficient Training Options

1. **Gradient Checkpointing**: `--gradient_checkpointing true`
2. **4-bit Quantization**: `--use_4bit true`
3. **DeepSpeed ZeRO-3**: `--deepspeed configs/deepspeed_zero3.json`
4. **Smaller Batch Size**: `--per_device_train_batch_size 1`
5. **Higher Gradient Accumulation**: `--gradient_accumulation_steps 32`

## 📈 Monitoring Training

### Weights & Biases

Training metrics are automatically logged to W&B. View your runs at:
https://wandb.ai/your-username/zizogpt

### TensorBoard

```bash
tensorboard --logdir ./outputs/pretrain
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--per_device_train_batch_size 1`
2. Enable gradient checkpointing: `--gradient_checkpointing true`
3. Use 4-bit quantization: `--use_4bit true`
4. Use DeepSpeed: `--deepspeed configs/deepspeed_zero3.json`
5. Reduce sequence length: `--max_seq_length 1024`

### Slow Data Loading

1. Use streaming mode for large datasets: `--streaming true`
2. Increase number of workers: `--dataloader_num_workers 8`
3. Enable pin memory: Enabled by default

### HuggingFace Authentication

```bash
huggingface-cli login
# Enter your access token from https://huggingface.co/settings/tokens
```

## 📚 References

- [NVIDIA Nemotron Paper](https://arxiv.org/abs/2508.14444)
- [Nemotron Pre-Training Datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets)
- [Nemotron Post-Training v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- [DeepSpeed](https://www.deepspeed.ai/)

## 📄 License

This project is for educational purposes. Please refer to individual dataset licenses on HuggingFace.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

**Happy Training! 🎉**
