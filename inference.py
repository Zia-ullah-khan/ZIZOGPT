"""
Inference Script for ZIZOGPT
Generate text using your trained model
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    GenerationConfig,
)
from peft import PeftModel


def load_model(
    model_path: str,
    lora_path: str = None,
    use_4bit: bool = False,
    device_map: str = "auto",
):
    """Load model for inference."""
    
    print(f"Loading model from {model_path}...")
    
    # Quantization config
    quantization_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load LoRA weights if provided
    if lora_path:
        print(f"Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stream: bool = True,
):
    """Generate text from prompt."""
    
    # Format prompt with chat template
    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Setup streamer
    streamer = TextStreamer(tokenizer, skip_special_tokens=True) if stream else None
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            streamer=streamer,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|assistant|>" in formatted_prompt:
        # Remove the prompt portion
        response_start = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
        response = generated_text[response_start:].strip()
    else:
        response = generated_text
    
    return response


def chat(model, tokenizer, **gen_kwargs):
    """Interactive chat mode."""
    
    print("\n" + "=" * 50)
    print("ZIZOGPT Chat Mode")
    print("Type 'quit' or 'exit' to end the conversation")
    print("=" * 50 + "\n")
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        conversation_history.append({"role": "user", "content": user_input})
        
        # Build prompt from history
        prompt = ""
        for msg in conversation_history:
            if msg["role"] == "user":
                prompt += f"<|user|>\n{msg['content']}<|end|>\n"
            else:
                prompt += f"<|assistant|>\n{msg['content']}<|end|>\n"
        prompt += "<|assistant|>\n"
        
        print("\nZIZOGPT: ", end="", flush=True)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 512),
                temperature=gen_kwargs.get("temperature", 0.7),
                top_p=gen_kwargs.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        conversation_history.append({"role": "assistant", "content": response.strip()})


def main():
    parser = argparse.ArgumentParser(description="ZIZOGPT Inference")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./outputs/rl/final",
        help="Path to the model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights (optional)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Enable interactive chat mode"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        lora_path=args.lora_path,
        use_4bit=args.use_4bit,
    )
    
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    if args.chat:
        chat(model, tokenizer, **gen_kwargs)
    elif args.prompt:
        response = generate(model, tokenizer, args.prompt, **gen_kwargs)
        print(f"\nResponse:\n{response}")
    else:
        # Default: interactive single-prompt mode
        print("\nEnter your prompt (press Enter twice to generate):")
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                break
        
        prompt = "\n".join(lines)
        if prompt:
            print("\nGenerating...\n")
            response = generate(model, tokenizer, prompt, **gen_kwargs)


if __name__ == "__main__":
    main()
