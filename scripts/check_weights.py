import torch
import os
import sys
import numpy as np
from pathlib import Path

# Try importing safetensors, handle if not installed
try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

def check_weights(model_path):
    print(f"==================================================")
    print(f"Weight Diagnostic for: {model_path}")
    print(f"==================================================")

    path_obj = Path(model_path)
    if not path_obj.exists():
        print(f"❌ Error: Path {model_path} does not exist.")
        return

    # 1. Identify Model Files
    files = list(path_obj.glob("*.safetensors"))
    is_safetensors = True
    if not files:
        files = list(path_obj.glob("pytorch_model*.bin"))
        is_safetensors = False
        if not files:
            # Check for adapter weights
            files = list(path_obj.glob("adapter_model.bin"))
            if not files:
                print("❌ No model weight files (.safetensors or .bin) found.")
                return

    print(f"Found {len(files)} weight file(s).")
    
    total_params = 0
    nan_layers = []
    zero_std_layers = []
    suspiciously_small_std_layers = [] # Potential collapse
    
    # 2. Iterate and Analyze
    for file_path in files:
        print(f"\nAnalyzing file: {file_path.name}...")
        
        try:
            if is_safetensors and HAS_SAFETENSORS:
                state_dict = load_safetensors(file_path)
            else:
                state_dict = torch.load(file_path, map_location="cpu")
        except Exception as e:
            print(f"❌ Failed to load file: {e}")
            continue

        for name, tensor in state_dict.items():
            # Skip integer tensors (buffers etc)
            if not torch.is_floating_point(tensor):
                continue
                
            tensor = tensor.float() # Convert to float32 for stats
            numel = tensor.numel()
            total_params += numel
            
            # Checks
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            std_dev = tensor.std().item()
            mean_val = tensor.mean().item()
            
            # Reporting
            status = "✅"
            if has_nan or has_inf:
                status = "❌ NAN/INF"
                nan_layers.append(name)
            elif std_dev == 0.0:
                 status = "⚠️ DEAD (STD=0)"
                 zero_std_layers.append(name)
            elif std_dev < 1e-6:
                 status = "⚠️ COLLAPSED?"
                 suspiciously_small_std_layers.append(name)

            # Print sample for first few or errors
            if has_nan or has_inf or std_dev < 1e-6 or "embed" in name or "layers.0" in name:
                print(f"  {status} {name: <50} | Mean: {mean_val:.6f} | Std: {std_dev:.8f} | Shape: {tuple(tensor.shape)}")

    # 3. Final Summary
    print(f"\n==================================================")
    print(f"SUMMARY REPORT")
    print(f"==================================================")
    print(f"Total Parameters Scanned: {total_params / 1e9:.3f} B")
    
    if nan_layers:
        print(f"\n❌ CRITICAL: Found {len(nan_layers)} layers with NaNs or Infs!")
        print(f"   Examples: {nan_layers[:5]}")
        print("   -> The model has mathematically exploded.")
    else:
        print(f"\n✅ No NaNs or Infs found.")

    if suspiciously_small_std_layers:
        print(f"\n⚠️ WARNING: Found {len(suspiciously_small_std_layers)} layers with extremely low variance (Std < 1e-6).")
        print(f"   Examples: {suspiciously_small_std_layers[:5]}")
        print("   -> This often indicates vanishing gradients or initialization failure.")
    
    if total_params > 0 and not nan_layers and not suspiciously_small_std_layers:
        print("\n✅ Weight statistics look physically healthy (no blatant corruption).")
    
    print(f"==================================================")

if __name__ == "__main__":
    target_dir = "./outputs/rl/final"
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    
    check_weights(target_dir)