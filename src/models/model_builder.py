"""
Model Builder Module for ZIZOGPT
Handles model initialization, configuration, and loading
"""

import os
from typing import Dict, Optional, Union
from dataclasses import dataclass

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
    MistralConfig,
    MistralForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
)
from peft import (
    LoraConfig,
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
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    initializer_range: float = 0.02
    use_flash_attention: bool = True
    pretrained_model_name_or_path: Optional[str] = None


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class ModelBuilder:
    """
    Builder class for creating and configuring LLM models.
    """
    
    ARCHITECTURE_MAP = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "mistral": (MistralConfig, MistralForCausalLM),
        "gpt2": (GPT2Config, GPT2LMHeadModel),
    }
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        
    def build_model_from_scratch(self) -> PreTrainedModel:
        """
        Build a new model from scratch using the specified configuration.
        
        Returns:
            Initialized model
        """
        logger.info(f"Building {self.model_config.architecture} model from scratch...")
        
        config_class, model_class = self.ARCHITECTURE_MAP.get(
            self.model_config.architecture,
            (LlamaConfig, LlamaForCausalLM)
        )
        
        # Create config based on architecture
        if self.model_config.architecture in ["llama", "mistral"]:
            config = config_class(
                hidden_size=self.model_config.hidden_size,
                intermediate_size=self.model_config.intermediate_size,
                num_hidden_layers=self.model_config.num_hidden_layers,
                num_attention_heads=self.model_config.num_attention_heads,
                num_key_value_heads=self.model_config.num_key_value_heads,
                vocab_size=self.model_config.vocab_size,
                max_position_embeddings=self.model_config.max_position_embeddings,
                rope_theta=self.model_config.rope_theta,
                initializer_range=self.model_config.initializer_range,
                use_cache=True,
                tie_word_embeddings=False,
            )
            
            # Enable flash attention if supported
            if self.model_config.use_flash_attention:
                config._attn_implementation = "flash_attention_2"
                
        elif self.model_config.architecture == "gpt2":
            config = config_class(
                n_embd=self.model_config.hidden_size,
                n_layer=self.model_config.num_hidden_layers,
                n_head=self.model_config.num_attention_heads,
                vocab_size=self.model_config.vocab_size,
                n_positions=self.model_config.max_position_embeddings,
            )
        else:
            raise ValueError(f"Unknown architecture: {self.model_config.architecture}")
        
        # Initialize model
        model = model_class(config)
        
        # Initialize weights
        model.apply(self._init_weights)
        
        # Calculate and log model size
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {num_params / 1e9:.2f}B parameters")
        
        return model
    
    def load_pretrained_model(
        self,
        model_name_or_path: str,
        use_4bit: bool = False,
        use_8bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ) -> PreTrainedModel:
        """
        Load a pretrained model from HuggingFace or local path.
        
        Args:
            model_name_or_path: Model identifier or path
            use_4bit: Use 4-bit quantization
            use_8bit: Use 8-bit quantization
            device_map: Device mapping strategy
            torch_dtype: Data type for model weights
            trust_remote_code: Trust remote code in model
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading pretrained model: {model_name_or_path}")
        
        # Setup quantization config if needed
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2" if self.model_config.use_flash_attention else None,
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {num_params / 1e9:.2f}B parameters")
        
        return model
    
    def add_lora(
        self,
        model: PreTrainedModel,
        lora_config: LoRAConfig,
        prepare_for_kbit: bool = False,
    ) -> PreTrainedModel:
        """
        Add LoRA adapters to the model.
        
        Args:
            model: Base model
            lora_config: LoRA configuration
            prepare_for_kbit: Prepare model for k-bit training
            
        Returns:
            Model with LoRA adapters
        """
        logger.info("Adding LoRA adapters to model...")
        
        if prepare_for_kbit:
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, peft_config)
        
        # Log trainable parameters
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA added: {trainable_params:,} trainable parameters "
            f"({100 * trainable_params / all_params:.2f}% of {all_params:,} total)"
        )
        
        return model
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        std = self.model_config.initializer_range
        
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


def load_tokenizer(
    model_name_or_path: str,
    model_max_length: int = 4096,
    padding_side: str = "right",
    use_fast: bool = True,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizer:
    """
    Load tokenizer from HuggingFace or local path.
    
    Args:
        model_name_or_path: Tokenizer identifier or path
        model_max_length: Maximum sequence length
        padding_side: Side to pad on
        use_fast: Use fast tokenizer
        trust_remote_code: Trust remote code
        
    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer: {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    
    # Add special tokens if needed
    special_tokens = {
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|end|>",
    }
    
    # Only add tokens that don't exist
    tokens_to_add = {}
    for key, value in special_tokens.items():
        if getattr(tokenizer, key, None) is None:
            tokens_to_add[key] = value
    
    if tokens_to_add:
        tokenizer.add_special_tokens(tokens_to_add)
        logger.info(f"Added special tokens: {tokens_to_add}")
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def create_model_and_tokenizer(
    config: Union[Dict, ModelConfig],
    tokenizer_path: Optional[str] = None,
    from_scratch: bool = False,
    use_lora: bool = False,
    lora_config: Optional[LoRAConfig] = None,
    use_4bit: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Create model and tokenizer based on configuration.
    
    Args:
        config: Model configuration
        tokenizer_path: Path to tokenizer (if different from model)
        from_scratch: Build model from scratch
        use_lora: Add LoRA adapters
        lora_config: LoRA configuration
        use_4bit: Use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if isinstance(config, dict):
        config = ModelConfig(**config)
    
    builder = ModelBuilder(config)
    
    # Load or create model
    if from_scratch:
        model = builder.build_model_from_scratch()
        tokenizer_source = tokenizer_path or "meta-llama/Llama-2-7b-hf"
    else:
        model_path = config.pretrained_model_name_or_path
        if not model_path:
            raise ValueError("pretrained_model_name_or_path required when not training from scratch")
        
        model = builder.load_pretrained_model(
            model_path,
            use_4bit=use_4bit,
        )
        tokenizer_source = tokenizer_path or model_path
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_source)
    
    # Resize embeddings if needed
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")
    
    # Add LoRA if requested
    if use_lora:
        if lora_config is None:
            lora_config = LoRAConfig()
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
