"""
Reinforcement Learning Training Script for ZIZOGPT
Implements DPO, PPO, and GRPO for preference alignment
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import datasets
from datasets import load_dataset, Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
        default="./outputs/sft/final",
        metadata={"help": "Path to the SFT model"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer path if different from model"}
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
        metadata={"help": "Use LoRA"}
    )
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "LoRA target modules"}
    )
    
    # Quantization
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Use 4-bit quantization"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    
    dataset_name: str = field(
        default="nvidia/Nemotron-3-Nano-RL-Training-Blend",
        metadata={"help": "RL training dataset"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Maximum prompt length"}
    )
    max_response_length: int = field(
        default=1024,
        metadata={"help": "Maximum response length"}
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes for preprocessing"}
    )


@dataclass
class RLTrainingArguments:
    """Training arguments for RL."""
    
    # Algorithm selection
    rl_algorithm: str = field(
        default="dpo",
        metadata={"help": "RL algorithm: dpo, ppo, grpo"}
    )
    
    # DPO specific
    beta: float = field(
        default=0.1,
        metadata={"help": "DPO beta (KL penalty coefficient)"}
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={"help": "DPO loss type: sigmoid, hinge, ipo"}
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Label smoothing for DPO"}
    )
    
    # Training
    output_dir: str = field(default="./outputs/rl")
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-7)
    num_train_epochs: float = field(default=1.0)
    max_steps: int = field(default=-1)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.0)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=100)
    save_total_limit: int = field(default=3)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    max_grad_norm: float = field(default=1.0)
    seed: int = field(default=42)
    
    # Logging
    report_to: str = field(default="wandb")
    run_name: str = field(default="zizogpt-rl")
    
    overwrite_output_dir: bool = field(default=True)


def prepare_dpo_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_prompt_length: int = 1024,
    num_proc: int = 4,
) -> Dataset:
    """
    Prepare dataset for DPO training.
    Expected format: prompt, chosen, rejected
    """
    
    def extract_dpo_pairs(examples: Dict[str, Any]) -> Dict[str, List]:
        prompts = []
        chosen_responses = []
        rejected_responses = []
        
        # Handle different formats
        if "chosen" in examples and "rejected" in examples:
            # Standard DPO format
            for i in range(len(examples["chosen"])):
                prompt = examples.get("prompt", [""] * len(examples["chosen"]))[i]
                chosen = examples["chosen"][i]
                rejected = examples["rejected"][i]
                
                # Handle nested conversation format
                if isinstance(chosen, list):
                    # Extract from conversation
                    chosen = format_conversation(chosen)
                    rejected = format_conversation(rejected)
                    prompt = extract_prompt(examples["chosen"][i])
                
                prompts.append(prompt)
                chosen_responses.append(chosen)
                rejected_responses.append(rejected)
                
        elif "response_a" in examples and "response_b" in examples:
            # A/B comparison format
            for i in range(len(examples["response_a"])):
                prompt = examples["prompt"][i]
                resp_a = examples["response_a"][i]
                resp_b = examples["response_b"][i]
                
                # Determine which is chosen based on preference
                pref = examples.get("preference", [1] * len(examples["response_a"]))[i]
                
                if pref == 1:  # A is preferred
                    chosen = resp_a
                    rejected = resp_b
                else:  # B is preferred
                    chosen = resp_b
                    rejected = resp_a
                
                prompts.append(prompt)
                chosen_responses.append(chosen)
                rejected_responses.append(rejected)
        
        return {
            "prompt": prompts,
            "chosen": chosen_responses,
            "rejected": rejected_responses,
        }
    
    def format_conversation(conversation: List[Dict]) -> str:
        """Format conversation to string."""
        if isinstance(conversation, str):
            return conversation
        
        formatted = ""
        for turn in conversation:
            role = turn.get("role", turn.get("from", "assistant"))
            content = turn.get("content", turn.get("value", ""))
            if role in ["assistant", "gpt", "bot"]:
                formatted += content
        return formatted
    
    def extract_prompt(conversation: List[Dict]) -> str:
        """Extract prompt from conversation."""
        if isinstance(conversation, str):
            return ""
        
        prompt_parts = []
        for turn in conversation:
            role = turn.get("role", turn.get("from", "user"))
            content = turn.get("content", turn.get("value", ""))
            if role in ["user", "human", "system"]:
                prompt_parts.append(f"<|{role}|>\n{content}<|end|>\n")
        return "".join(prompt_parts)
    
    # Process dataset
    processed = dataset.map(
        extract_dpo_pairs,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    
    # Filter out empty samples
    processed = processed.filter(
        lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0
    )
    
    return processed


def main():
    """Main RL training function."""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, RLTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(training_args.seed)
    
    # Setup logging
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
    
    # Load reference model (for DPO)
    ref_model = None
    if training_args.rl_algorithm == "dpo":
        ref_path = model_args.ref_model_name_or_path or model_args.model_name_or_path
        logger.info(f"Loading reference model: {ref_path}")
        
        if not model_args.use_lora:
            # Need separate reference model
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_path,
                quantization_config=quantization_config,
                device_map="auto" if quantization_config else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path or model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",  # DPO requires left padding
    )
    
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
    
    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # Load dataset
    logger.info(f"Loading RL dataset: {data_args.dataset_name}")
    
    try:
        raw_dataset = load_dataset(data_args.dataset_name, split="train")
    except Exception as e:
        logger.warning(f"Failed to load {data_args.dataset_name}: {e}")
        logger.info("Creating synthetic DPO dataset for testing...")
        
        # Create a small synthetic dataset for testing
        raw_dataset = Dataset.from_dict({
            "prompt": [
                "What is 2 + 2?",
                "Explain quantum computing.",
                "Write a haiku about coding.",
            ] * 100,
            "chosen": [
                "2 + 2 equals 4.",
                "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways classical computers cannot.",
                "Lines of code flow down\nBugs hide in the logic maze\nDebugger awaits",
            ] * 100,
            "rejected": [
                "I don't know.",
                "It's complicated.",
                "Poetry is hard.",
            ] * 100,
        })
    
    # Prepare dataset
    train_dataset = prepare_dpo_dataset(
        raw_dataset,
        tokenizer,
        max_prompt_length=data_args.max_prompt_length,
        num_proc=data_args.num_proc,
    )
    
    logger.info(f"Prepared {len(train_dataset)} training examples")
    
    if training_args.rl_algorithm == "dpo":
        # DPO Training
        logger.info("Starting DPO training...")
        
        dpo_config = DPOConfig(
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
            save_total_limit=training_args.save_total_limit,
            bf16=training_args.bf16,
            tf32=training_args.tf32,
            gradient_checkpointing=training_args.gradient_checkpointing,
            max_grad_norm=training_args.max_grad_norm,
            report_to=training_args.report_to,
            run_name=training_args.run_name,
            beta=training_args.beta,
            loss_type=training_args.loss_type,
            label_smoothing=training_args.label_smoothing,
            max_length=data_args.max_seq_length,
            max_prompt_length=data_args.max_prompt_length,
            remove_unused_columns=False,
        )
        
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        
    else:
        raise ValueError(f"Unknown RL algorithm: {training_args.rl_algorithm}")
    
    # Train
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save model
    logger.info("Saving final model...")
    final_path = os.path.join(training_args.output_dir, "final")
    
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("RL training complete!")
    logger.info(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
