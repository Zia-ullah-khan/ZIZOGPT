"""
Pre-training Script for ZIZOGPT
Train a language model from scratch using NVIDIA Nemotron datasets
"""

import os
import sys
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
import datasets
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_loader import NemotronDataLoader, DatasetConfig
from src.models.model_builder import (
    ModelBuilder,
    ModelConfig,
    create_model_and_tokenizer,
    load_tokenizer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from HuggingFace"}
    )
    architecture: str = field(
        default="llama",
        metadata={"help": "Model architecture: llama, mistral, gpt2"}
    )
    hidden_size: int = field(
        default=2048,
        metadata={"help": "Hidden size of the model"}
    )
    intermediate_size: int = field(
        default=5504,
        metadata={"help": "Intermediate size for MLP layers"}
    )
    num_hidden_layers: int = field(
        default=24,
        metadata={"help": "Number of transformer layers"}
    )
    num_attention_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads."}
    )
    num_key_value_heads: int = field(
        default=16,
        metadata={"help": "Number of key-value heads."}
    )
    vocab_size: int = field(
        default=128000,
        metadata={"help": "Vocabulary size of the model."}
    )
    max_position_embeddings: int = field(
        default=262144,
        metadata={"help": "Maximum sequence length."}
    )
    rope_theta: float = field(
        default=10000.0,
        metadata={"help": "RoPE theta value."}
    )
    initializer_range: float = field(
        default=0.02,
        metadata={"help": "Initializer range for model weights."}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Enable Flash Attention 2 for faster training."}
    )
    from_scratch: bool = field(
        default=False,
        metadata={"help": "Train model from scratch instead of loading pretrained"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer path if different from model (or path to custom trained tokenizer)"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    
    use_sample_dataset: bool = field(
        default=False,
        metadata={"help": "Use sample dataset for testing"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for training"}
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Use streaming mode for large datasets"}
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes for data preprocessing"}
    )
    dataset_weights: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated weights for datasets (e.g., '0.5,0.2,0.15,0.1,0.05')"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended training arguments."""
    
    # Override defaults
    output_dir: str = field(default="./outputs/pretrain")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=3e-4)
    num_train_epochs: float = field(default=1.0)
    max_steps: int = field(default=-1)
    warmup_steps: int = field(default=1000)
    weight_decay: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    
    # Distributed training
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file"}
    )
    local_rank: int = field(default=-1)
    
    # Logging
    report_to: str = field(default="wandb")
    run_name: Optional[str] = field(default="zizogpt-pretrain")


def get_pretrain_datasets(data_args: DataArguments, streaming: bool = True) -> List[DatasetConfig]:
    """Get list of pre-training dataset configurations."""
    
    if data_args.use_sample_dataset:
        return [
            DatasetConfig(
                name="nvidia/Nemotron-Pretraining-Dataset-sample",
                config_name="Nemotron-CC-High-Quality",
                weight=1.0,
                streaming=False,
                split="train",
            )
        ]
    
    # Parse custom weights if provided
    weights = [0.5, 0.2, 0.15, 0.1, 0.05]
    if data_args.dataset_weights:
        weights = [float(w) for w in data_args.dataset_weights.split(",")]
    
    datasets = [
        DatasetConfig(
            name="nvidia/Nemotron-CC-v2.1",
            weight=weights[0] if len(weights) > 0 else 0.5,
            streaming=streaming,
            split="train",
        ),
        DatasetConfig(
            name="nvidia/Nemotron-Pretraining-Code-v2",
            weight=weights[1] if len(weights) > 1 else 0.2,
            streaming=streaming,
            split="train",
        ),
        DatasetConfig(
            name="nvidia/Nemotron-CC-Math-v1",
            weight=weights[2] if len(weights) > 2 else 0.15,
            streaming=streaming,
            split="train",
        ),
        DatasetConfig(
            name="nvidia/Nemotron-Pretraining-Specialized-v1",
            weight=weights[3] if len(weights) > 3 else 0.1,
            streaming=streaming,
            split="train",
        ),
        DatasetConfig(
            name="nvidia/Nemotron-CC-Code-v1",
            weight=weights[4] if len(weights) > 4 else 0.05,
            streaming=streaming,
            split="train",
        ),
    ]
    
    return datasets


def main():
    """Main training function."""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    
    # Robustly find config file in args
    config_file = None
    for arg in sys.argv[1:]:
        if arg.endswith(".json") or arg.endswith(".yaml") or arg.endswith(".yml"):
            config_file = arg
            break
            
    if config_file and config_file.endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(config_file)
    elif config_file and (config_file.endswith(".yaml") or config_file.endswith(".yml")):
        model_args, data_args, training_args = parser.parse_yaml_file(config_file)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Setup logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    # Log configuration
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming from {last_checkpoint}")
    
    # Create model and tokenizer
    if model_args.from_scratch:
        logger.info("Building model from scratch...")
        model_config = ModelConfig(
            architecture=model_args.architecture,
            hidden_size=model_args.hidden_size,
            intermediate_size=model_args.intermediate_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            num_key_value_heads=model_args.num_key_value_heads,
            vocab_size=model_args.vocab_size,
            max_position_embeddings=model_args.max_position_embeddings,
            rope_theta=model_args.rope_theta,
            initializer_range=model_args.initializer_range,
            use_flash_attention_2=model_args.use_flash_attention_2,
        )
        model, tokenizer = create_model_and_tokenizer(
            model_config=model_config,
            from_scratch=True,
            tokenizer_path=model_args.tokenizer_path
        )
    else:
        logger.info("Loading model from pretrained checkpoint...")
        model_config = ModelConfig(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            use_flash_attention_2=model_args.use_flash_attention_2,
        )
        model, tokenizer = create_model_and_tokenizer(
            model_config=model_config,
            tokenizer_path=model_args.tokenizer_path,
            from_scratch=False,
        )
    
    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # Load datasets
    logger.info("Loading pre-training datasets...")
    data_loader = NemotronDataLoader(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        num_proc=data_args.num_proc,
    )
    
    dataset_configs = get_pretrain_datasets(data_args, streaming=data_args.streaming)
    train_dataset = data_loader.load_pretrain_datasets(
        dataset_configs=dataset_configs
    )
    
    logger.info(f"Dataset loaded: {train_dataset}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    
    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(training_args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "final"))
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("Training complete!")
    logger.info(f"Model saved to {training_args.output_dir}/final")


if __name__ == "__main__":
    main()
