"""
Data Loading Module for ZIZOGPT Training
Handles loading and preprocessing of NVIDIA Nemotron datasets
"""

import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from functools import partial

import torch
from datasets import load_dataset, interleave_datasets, IterableDataset, Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    weight: float = 1.0
    streaming: bool = True
    split: str = "train"
    config_name: Optional[str] = None  # Added for Nemotron-like datasets


class NemotronDataLoader:
    """
    Data loader for NVIDIA Nemotron Pre-Training and Post-Training datasets.
    """
    
    # Pre-training datasets from NVIDIA
    PRETRAIN_DATASETS = {
        "cc_v2.1": "nvidia/Nemotron-CC-v2.1",
        "cc_v2": "nvidia/Nemotron-CC-v2",
        "code_v2": "nvidia/Nemotron-Pretraining-Code-v2",
        "code_v1": "nvidia/Nemotron-Pretraining-Code-v1",
        "cc_code_v1": "nvidia/Nemotron-CC-Code-v1",
        "math_v1": "nvidia/Nemotron-CC-Math-v1",
        "specialized_v1": "nvidia/Nemotron-Pretraining-Specialized-v1",
        "sft_v1": "nvidia/Nemotron-Pretraining-SFT-v1",
        "sample": "nvidia/Nemotron-Pretraining-Dataset-sample",
    }
    
    # Post-training datasets from NVIDIA
    POSTTRAIN_DATASETS = {
        "instruction_chat": "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "math_v2": "nvidia/Nemotron-Math-v2",
        "math_proofs": "nvidia/Nemotron-Math-Proofs-v1",
        "science": "nvidia/Nemotron-Science-v1",
        "agentic": "nvidia/Nemotron-Agentic-v1",
        "competitive_programming": "nvidia/Nemotron-Competitive-Programming-v1",
        "rl_blend": "nvidia/Nemotron-3-Nano-RL-Training-Blend",
    }
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        num_proc: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_proc = num_proc
        
    def load_pretrain_datasets(
        self,
        dataset_configs: List[DatasetConfig],
    ) -> Union[IterableDataset, Dataset]:
        """
        Load and combine pre-training datasets.
        
        Args:
            dataset_configs: List of DatasetConfig objects
            
        Returns:
            Combined dataset (iterable or regular)
        """
        loaded_datasets = []
        probabilities = []

        for config in dataset_configs:
            try:
                logger.info(f"Loading pretraining dataset: {config.name} with config: {config.config_name}")
                ds = load_dataset(
                    config.name,
                    name=config.config_name,
                    split=config.split,
                    streaming=config.streaming
                )
                loaded_datasets.append(ds)
                probabilities.append(config.weight)
            except Exception as e:
                logger.warning(f"Failed to load dataset {config.name}: {e}")
                continue

        if not loaded_datasets:
            raise ValueError("No pretraining datasets could be loaded. Please check your config.")

        # Normalize weights
        total_weight = sum(probabilities)
        probabilities = [p / total_weight for p in probabilities]

        logger.info(f"Interleaving {len(loaded_datasets)} datasets with weights {probabilities}")
        if len(loaded_datasets) > 1:
            # Use IterableDataset for streaming
            raw_datasets = interleave_datasets(loaded_datasets, probabilities=probabilities, stopping_strategy="first_exhausted")
        else:
            raw_datasets = loaded_datasets[0]

        def tokenize_function(examples: Dict[str, Any]) -> Dict[str, List[int]]:
            # Dynamically find the text column
            text_column = next((k for k in ["content", "text", "code", "input"] if k in examples), "text")
            
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_token_type_ids=False
            )

        logger.info("Tokenizing pretraining datasets...")
        # For streaming, map is lazy. For non-streaming, we can use num_proc
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=list(raw_datasets.features.keys()) if hasattr(raw_datasets, 'features') else None
        )
        
        # Select necessary columns for IterableDataset
        if isinstance(tokenized_datasets, IterableDataset):
             tokenized_datasets = tokenized_datasets.select_columns(["input_ids", "attention_mask"])

        return tokenized_datasets
    
    def load_sft_datasets(
        self,
        dataset_configs: List[DatasetConfig],
    ) -> Union[IterableDataset, Dataset]:
        """
        Load and combine SFT datasets.
        
        Args:
            dataset_configs: List of DatasetConfig objects
            
        Returns:
            Combined dataset for supervised fine-tuning
        """
        loaded_datasets = []
        probabilities = []

        for config in dataset_configs:
            try:
                logger.info(f"Loading SFT dataset: {config.name}")
                ds = load_dataset(config.name, split=config.split, streaming=config.streaming)
                loaded_datasets.append(ds)
                probabilities.append(config.weight)
            except Exception as e:
                logger.warning(f"Failed to load SFT dataset {config.name}: {e}")
                continue

        if not loaded_datasets:
            raise ValueError("No SFT datasets could be loaded.")

        total_weight = sum(probabilities)
        probabilities = [p / total_weight for p in probabilities]

        if len(loaded_datasets) > 1:
            raw_datasets = interleave_datasets(loaded_datasets, probabilities=probabilities, stopping_strategy="first_exhausted")
        else:
            raw_datasets = loaded_datasets[0]

        def format_chat(example: Dict[str, Any]) -> Dict[str, str]:
            # This function attempts to handle various chat formats
            formatted_text = ""
            if "conversations" in example:
                for turn in example["conversations"]:
                    role = turn.get("from", "user").lower()
                    content = turn.get("value", "")
                    formatted_text += f"<|{role}|>\n{content}\n"
            elif "messages" in example:
                for msg in example["messages"]:
                    role = msg.get("role", "user").lower()
                    content = msg.get("content", "")
                    formatted_text += f"<|{role}|>\n{content}\n"
            elif "prompt" in example and "response" in example:
                formatted_text = f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}\n"
            elif "instruction" in example and "output" in example:
                instruction = example['instruction']
                input_text = example.get('input', '')
                prompt = f"{instruction}\n{input_text}".strip()
                formatted_text = f"<|user|>\n{prompt}\n<|assistant|>\n{example['output']}\n"
            elif "text" in example:
                 formatted_text = example["text"] # Assumes pre-formatted
            else:
                 # Fallback for unknown formats
                 formatted_text = str(example)
            
            return {"text": formatted_text}

        logger.info("Formatting SFT datasets...")
        # Use a consistent 'text' column for the SFTTrainer
        # Remove original columns to prevent SFTTrainer from trying to apply chat template
        column_names = list(next(iter(raw_datasets)).keys())
        formatted_dataset = raw_datasets.map(
            format_chat,
            remove_columns=column_names
        )
        
        return formatted_dataset
    
    def load_rl_dataset(
        self,
        dataset_name: str = "nvidia/Nemotron-3-Nano-RL-Training-Blend",
        split: str = "train",
    ) -> Dataset:
        """
        Load RL training dataset.
        
        Args:
            dataset_name: Name of the RL dataset
            split: Dataset split to load
            
        Returns:
            Dataset for RL training
        """
        logger.info(f"Loading RL dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        return self._preprocess_rl(dataset)
    
    def _preprocess_rl(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset for RL training."""
        
        def format_rl_example(examples):
            prompts = []
            chosen = []
            rejected = []
            
            # Handle DPO-style data
            if "chosen" in examples and "rejected" in examples:
                for prompt, ch, rej in zip(
                    examples.get("prompt", [""] * len(examples["chosen"])),
                    examples["chosen"],
                    examples["rejected"]
                ):
                    prompts.append(prompt)
                    chosen.append(ch)
                    rejected.append(rej)
            # Handle preference data
            elif "prompt" in examples and "response_a" in examples:
                for prompt, resp_a, resp_b, pref in zip(
                    examples["prompt"],
                    examples["response_a"],
                    examples["response_b"],
                    examples.get("preference", [1] * len(examples["prompt"]))
                ):
                    prompts.append(prompt)
                    if pref == 1:
                        chosen.append(resp_a)
                        rejected.append(resp_b)
                    else:
                        chosen.append(resp_b)
                        rejected.append(resp_a)
            
            return {
                "prompt": prompts,
                "chosen": chosen,
                "rejected": rejected,
            }
        
        return dataset.map(
            format_rl_example,
            batched=True,
            num_proc=self.num_proc,
        )
    

def create_data_collator(tokenizer: PreTrainedTokenizer, max_length: int = 2048):
    """Create a data collator for language modeling."""
    from transformers import DataCollatorForLanguageModeling
    
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8,
    )


def get_dataloader(
    dataset: Union[IterableDataset, Dataset],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    max_length: int = 2048,
) -> DataLoader:
    """Create a DataLoader from a dataset."""
    
    collator = create_data_collator(tokenizer, max_length)
    
    if isinstance(dataset, IterableDataset):
        shuffle = False  # Can't shuffle iterable datasets
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test the data loader
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    loader = NemotronDataLoader(tokenizer, max_seq_length=512)
    
    # Test with sample dataset
    sample_config = [
        DatasetConfig(name="nvidia/Nemotron-Pretraining-Dataset-sample", streaming=False)
    ]
    
    print("Loading sample dataset...")
    dataset = loader.load_pretrain_datasets(sample_config)
    print(f"Dataset loaded: {dataset}")
    print(f"Sample: {next(iter(dataset))}")
