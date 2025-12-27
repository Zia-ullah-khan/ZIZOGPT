import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import json

def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from scratch")
    parser.add_argument("--vocab_size", type=int, default=128000, help="Vocabulary size")
    parser.add_argument("--output_dir", type=str, default="./tokenizer", help="Directory to save the tokenizer")
    parser.add_argument("--sample_size", type=int, default=1000000, help="Number of samples to use for training from the sample dataset")
    args = parser.parse_args()

    print(f"Training tokenizer with vocab size {args.vocab_size}")
    print(f"Saving to {args.output_dir}")

    # For training the tokenizer, we can use a sample of the data.
    # The sample dataset is good for this.
    dataset = load_dataset("nvidia/Nemotron-Pretraining-Dataset-sample", "Nemotron-CC-High-Quality", split="train")

    def text_iterator(sample_size):
        count = 0
        for item in dataset:
            if count >= sample_size:
                break
            yield item['text']
            count += 1
    
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train_from_iterator(
        text_iterator(args.sample_size),
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=[
            "<|pad|>",
            "<|bos|>",
            "<|end|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>"
        ],
    )

    # Save tokenizer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Also save a tokenizer_config.json for AutoTokenizer
    config = {
        "model_type": "gpt2",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|end|>",
        "unk_token": "<unk>"
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()
