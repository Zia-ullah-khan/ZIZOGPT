"""
Supervised Fine-Tuning (SFT) Script for ZIZOGPT
Fine-tune a pre-trained model using NVIDIA Nemotron post-training datasets
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_loader import NemotronDataLoader, DatasetConfig
from src.models.model_builder import load_tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    model_name_or_path: str = field(
        default="./outputs/pretrain/final",
        metadata={"help": "Path to pretrained model"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name or path if different from model"}
    )
    num_attention_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads."}
    )
    num_key_value_heads: int = field(
        default=16,
        metadata={"help": "Number of key-value heads."}
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
    
    # LoRA configuration
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for efficient fine-tuning"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    
    # Quantization
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit quantization"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    
    max_seq_length: int = field(
        default=4096,
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
    dataset_subset: Optional[str] = field(
        default=None,
        metadata={"help": "Subset of datasets to use: all, math, code, science, chat"}
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Pack multiple samples into single sequence"}
    )


@dataclass
class SFTTrainingArguments(TrainingArguments):
    """Extended training arguments for SFT."""
    
    output_dir: str = field(default="./outputs/sft")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-5)
    num_train_epochs: float = field(default=3.0)
    max_steps: int = field(default=-1)
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.0)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=250)
    save_total_limit: int = field(default=3)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    
    # Distributed training
    deepspeed: Optional[str] = field(default=None)
    local_rank: int = field(default=-1)
    
    # Logging
    report_to: str = field(default="wandb")
    run_name: Optional[str] = field(default="zizogpt-sft")


def get_sft_datasets(data_args: DataArguments, streaming: bool = True) -> List[DatasetConfig]:
    """Get list of SFT dataset configurations."""
    
    all_datasets = {
        "chat": DatasetConfig(
            name="nvidia/Nemotron-Instruction-Following-Chat-v1",
            weight=0.3,
            streaming=streaming,
            split="train",
        ),
        "math": DatasetConfig(
            name="nvidia/Nemotron-Math-v2",
            weight=0.25,
            streaming=streaming,
            split="train",
        ),
        "math_proofs": DatasetConfig(
            name="nvidia/Nemotron-Math-Proofs-v1",
            weight=0.15,
            streaming=streaming,
            split="train",
        ),
        "science": DatasetConfig(
            name="nvidia/Nemotron-Science-v1",
            weight=0.15,
            streaming=streaming,
            split="train",
        ),
        "agentic": DatasetConfig(
            name="nvidia/Nemotron-Agentic-v1",
            weight=0.1,
            streaming=False,  # Small dataset
            split="train",
        ),
        "code": DatasetConfig(
            name="nvidia/Nemotron-Competitive-Programming-v1",
            weight=0.05,
            streaming=False,  # Small dataset
            split="train",
        ),
    }
    
    subset = data_args.dataset_subset
    
    if subset is None or subset == "all":
        return list(all_datasets.values())
    elif subset == "math":
        return [all_datasets["math"], all_datasets["math_proofs"]]
    elif subset == "code":
        return [all_datasets["code"]]
    elif subset == "science":
        return [all_datasets["science"]]
    elif subset == "chat":
        return [all_datasets["chat"]]
    else:
        logger.warning(f"Unknown subset: {subset}, using all datasets")
        return list(all_datasets.values())


def format_chat_template(example: Dict[str, Any]) -> str:
    """Format an example using chat template."""
    
    # Handle different data formats
    if "conversations" in example:
        formatted = ""
        for turn in example["conversations"]:
            role = turn.get("role", turn.get("from", "user"))
            content = turn.get("content", turn.get("value", ""))
            
            if role in ["user", "human"]:
                formatted += f"<|user|>\n{content}<|end|>\n"
            elif role in ["assistant", "gpt", "bot"]:
                formatted += f"<|assistant|>\n{content}<|end|>\n"
            elif role == "system":
                formatted += f"<|system|>\n{content}<|end|>\n"
        return formatted
        
    elif "messages" in example:
        formatted = ""
        for msg in example["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|{role}|>\n{content}<|end|>\n"
        return formatted
        
    elif "prompt" in example and "response" in example:
        prompt = example["prompt"]
        response = example["response"]
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>"
        
    elif "instruction" in example:
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example.get("output", example.get("response", ""))
        
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction
        
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{output}<|end|>"
        
    elif "text" in example:
        return example["text"]
    
    else:
        # Try to find any text-like field
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 10:
                return value
        
        raise ValueError(f"Could not format example with keys: {example.keys()}")


def main():
    """Main SFT function."""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(training_args.seed)
    
    # Setup logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")
    
    # Detect checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"Checkpoint detected: {last_checkpoint}")
    
    # Setup quantization
    quantization_config = None
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    
    if model_args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif model_args.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto" if quantization_config else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 else None,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        trust_remote_code=True,
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Setup LoRA
    peft_config = None
    if model_args.use_lora:
        logger.info("Setting up LoRA...")
        
        if quantization_config:
            model = prepare_model_for_kbit_training(model)
        
        target_modules = model_args.lora_target_modules.split(",")
        
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # Load datasets
    logger.info("Loading SFT datasets...")
    dataset_configs = get_sft_datasets(data_args, streaming=data_args.streaming)
    
    data_loader = NemotronDataLoader(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        num_proc=data_args.num_proc,
    )
    
    train_dataset = data_loader.load_sft_datasets(dataset_configs)
    
    logger.info(f"Dataset loaded: {train_dataset}")
    
    # Create SFT config
    sft_config = SFTConfig(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        warmup_ratio=training_args.warmup_ratio,
        weight_decay=training_args.weight_decay,
        lr_scheduler_type=training_args.lr_scheduler_type,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps if training_args.eval_strategy != "no" else None,
        save_total_limit=training_args.save_total_limit,
        bf16=training_args.bf16,
        tf32=training_args.tf32,
        gradient_checkpointing=training_args.gradient_checkpointing,
        dataloader_num_workers=training_args.dataloader_num_workers,
        remove_unused_columns=training_args.remove_unused_columns,
        report_to=training_args.report_to,
        run_name=training_args.run_name,
        max_seq_length=data_args.max_seq_length,
        packing=data_args.packing,
        dataset_text_field="text",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config if not model_args.use_lora else None,  # Already applied
    )
    
    # Train
    logger.info("Starting SFT training...")
    
    checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save model
    logger.info("Saving final model...")
    final_path = os.path.join(training_args.output_dir, "final")
    
    if model_args.use_lora:
        # Save LoRA weights
        model.save_pretrained(final_path)
        
        # Optionally merge and save full model
        merged_path = os.path.join(training_args.output_dir, "merged")
        logger.info(f"Merging LoRA weights and saving to {merged_path}...")
        
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
    else:
        trainer.save_model(final_path)
    
    tokenizer.save_pretrained(final_path)
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("SFT training complete!")
    logger.info(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
