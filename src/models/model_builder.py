"""
Model Builder Module for ZIZOGPT
Handles model initialization, configuration, and loading
"""

import os
from typing import Dict, Optional, Union, List
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    LlamaConfig,
    LlamaForCausalLM,
)
from peft import (
    LoraConfig as PeftLoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    architecture: str = "llama"
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    vocab_size: int = 128000  # Updated
    max_position_embeddings: int = 262144  # Updated
    rope_theta: float = 10000.0
    initializer_range: float = 0.02
    use_flash_attention_2: bool = True # Corrected name
    pretrained_model_name_or_path: Optional[str] = None


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class ModelBuilder:
    """
    Builder class for creating and configuring LLM models.
    """
    
    ARCHITECTURE_MAP = {
        "llama": (LlamaConfig, LlamaForCausalLM),
    }

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def build_model_from_scratch(self) -> PreTrainedModel:
        """
        Build a new model from scratch using the specified configuration.
        """
        logger.info(f"Building {self.model_config.architecture} model from scratch...")

        config_class, model_class = self.ARCHITECTURE_MAP.get(
            self.model_config.architecture,
            (LlamaConfig, LlamaForCausalLM) # Default to Llama
        )

        config_kwargs = {
            "hidden_size": self.model_config.hidden_size,
            "intermediate_size": self.model_config.intermediate_size,
            "num_hidden_layers": self.model_config.num_hidden_layers,
            "num_attention_heads": self.model_config.num_attention_heads,
            "num_key_value_heads": self.model_config.num_key_value_heads,
            "vocab_size": self.model_config.vocab_size,
            "max_position_embeddings": self.model_config.max_position_embeddings,
            "rope_theta": self.model_config.rope_theta,
            "initializer_range": self.model_config.initializer_range,
            "use_cache": True,
            "tie_word_embeddings": False,
        }

        # Enable flash attention if specified
        if self.model_config.use_flash_attention_2:
            config_kwargs["attn_implementation"] = "flash_attention_2"

        config = config_class(**config_kwargs)

        # Initialize model
        logger.info(f"Initializing model with config: {config}")
        model = model_class(config)

        # Cast to bfloat16 if using flash attention to avoid warnings
        if self.model_config.use_flash_attention_2:
            logger.info("Converting model to bfloat16 for Flash Attention 2")
            model = model.to(torch.bfloat16)

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized from scratch with {num_params / 1e9:.2f}B parameters")

        return model

    def load_pretrained_model(
        self,
        use_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ) -> PreTrainedModel:
        """
        Load a pretrained model from HuggingFace or local path.
        """
        model_path = self.model_config.pretrained_model_name_or_path
        if not model_path:
            raise ValueError("`pretrained_model_name_or_path` must be set in ModelConfig to load a model.")

        logger.info(f"Loading pretrained model: {model_path}")

        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2" if self.model_config.use_flash_attention_2 else "eager",
        )

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Pretrained model loaded with {num_params / 1e9:.2f}B parameters")

        return model

    def add_lora(
        self,
        model: PreTrainedModel,
        lora_config: LoRAConfig,
        prepare_for_kbit: bool = False,
    ) -> PreTrainedModel:
        """
        Add LoRA adapters to the model.
        """
        logger.info("Adding LoRA adapters to the model...")

        if prepare_for_kbit:
            model = prepare_model_for_kbit_training(model)

        peft_config = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, peft_config)

        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA added: {trainable_params:,} trainable parameters "
            f"({100 * trainable_params / all_params:.2f}% of {all_params:,} total)"
        )

        return model


def load_tokenizer(
    tokenizer_path: str,
    model_max_length: int = 4096,
    padding_side: str = "right",
    use_fast: bool = True,
) -> PreTrainedTokenizer:
    """
    Load a tokenizer from a local path or HuggingFace hub.
    """
    logger.info(f"Loading tokenizer from: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=True, # Often needed for custom tokenizers
    )

    # Define standard special tokens
    special_tokens = {
        "pad_token": "<|pad|>",
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
    }

    # Add only if they don't exist
    tokens_to_add = {k: v for k, v in special_tokens.items() if getattr(tokenizer, f"{k}_id", None) is None}
    if tokens_to_add:
        tokenizer.add_special_tokens(tokens_to_add)
        logger.info(f"Added special tokens: {tokens_to_add}")

    # A pad token is essential for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("`pad_token` was not set, using `eos_token` as pad token.")

    return tokenizer


def create_model_and_tokenizer(
    model_config: ModelConfig,
    lora_config: Optional[LoRAConfig] = None,
    tokenizer_path: Optional[str] = None,
    from_scratch: bool = False,
    use_4bit: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    High-level function to create both model and tokenizer.
    """
    builder = ModelBuilder(model_config)

    # Determine the source for the tokenizer. If building from scratch,
    # a tokenizer path MUST be provided.
    if from_scratch:
        if not tokenizer_path:
            raise ValueError("`tokenizer_path` is required when `from_scratch=True`.")
        model = builder.build_model_from_scratch()
        tokenizer_source = tokenizer_path
    else:
        model_path = model_config.pretrained_model_name_or_path
        if not model_path:
            raise ValueError("`pretrained_model_name_or_path` is required when `from_scratch=False`.")
        
        model = builder.load_pretrained_model(use_4bit=use_4bit)
        tokenizer_source = tokenizer_path or model_path

    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_source, model_max_length=model_config.max_position_embeddings)

    # Align model vocab size with tokenizer vocab size
    if len(tokenizer) != model.config.vocab_size:
        logger.warning(
            f"Vocab size mismatch. Tokenizer has {len(tokenizer)} tokens, "
            f"model has {model.config.vocab_size}. Resizing model embeddings..."
        )
        model.resize_token_embeddings(len(tokenizer))
        # Also update the model's config to reflect the new size
        model.config.vocab_size = len(tokenizer)
        model_config.vocab_size = len(tokenizer)

    # Apply LoRA if configured
    if lora_config:
        model = builder.add_lora(model, lora_config, prepare_for_kbit=use_4bit)

    return model, tokenizer


if __name__ == "__main__":
    # Test model building
    config = ModelConfig(
        architecture="llama",
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=32000,
        max_position_embeddings=2048,
    )
    
    builder = ModelBuilder(config)
    model = builder.build_model_from_scratch()
    print(f"Built model: {model}")
    print(f"Config: {model.config}")
