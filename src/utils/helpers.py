"""
Utility Functions for ZIZOGPT Training
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

import torch
import torch.distributed as dist
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
) -> None:
    """Setup logging configuration."""
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to file."""
    
    path = Path(output_path)
    os.makedirs(path.parent, exist_ok=True)
    
    with open(path, "w") as f:
        if path.suffix in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == ".json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def get_model_size(model: PreTrainedModel) -> Dict[str, Any]:
    """Get model size information."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory footprint
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0,
        "param_size_mb": param_size / (1024 * 1024),
        "buffer_size_mb": buffer_size / (1024 * 1024),
        "total_size_mb": (param_size + buffer_size) / (1024 * 1024),
        "total_params_billions": total_params / 1e9,
    }


def print_model_size(model: PreTrainedModel, name: str = "Model") -> None:
    """Print model size information."""
    
    info = get_model_size(model)
    
    print(f"\n{'='*50}")
    print(f"{name} Size Information:")
    print(f"{'='*50}")
    print(f"Total parameters: {info['total_params']:,} ({info['total_params_billions']:.2f}B)")
    print(f"Trainable parameters: {info['trainable_params']:,} ({info['trainable_percent']:.2f}%)")
    print(f"Model size: {info['total_size_mb']:.2f} MB")
    print(f"{'='*50}\n")


def save_model_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """Save model checkpoint with optional metadata."""
    
    if step is not None:
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{timestamp}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save metadata
    metadata = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "model_size": get_model_size(model),
    }
    
    if metrics:
        metadata["metrics"] = metrics
    
    with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")
    return checkpoint_dir


def is_main_process() -> bool:
    """Check if current process is the main process."""
    
    if not dist.is_initialized():
        return True
    
    return dist.get_rank() == 0


def get_device() -> torch.device:
    """Get the appropriate device for training."""
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory information."""
    
    if not torch.cuda.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    
    for i in range(torch.cuda.device_count()):
        device_info = {
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
            "allocated_memory_gb": torch.cuda.memory_allocated(i) / (1024**3),
            "cached_memory_gb": torch.cuda.memory_reserved(i) / (1024**3),
        }
        info["devices"].append(device_info)
    
    return info


def print_gpu_info() -> None:
    """Print GPU information."""
    
    info = get_gpu_memory_info()
    
    if not info["available"]:
        print("No GPU available")
        return
    
    print(f"\n{'='*50}")
    print(f"GPU Information:")
    print(f"{'='*50}")
    print(f"Number of GPUs: {info['device_count']}")
    
    for i, device in enumerate(info["devices"]):
        print(f"\nGPU {i}: {device['name']}")
        print(f"  Total Memory: {device['total_memory_gb']:.2f} GB")
        print(f"  Allocated: {device['allocated_memory_gb']:.2f} GB")
        print(f"  Cached: {device['cached_memory_gb']:.2f} GB")
    
    print(f"{'='*50}\n")


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    time_per_step: float,  # seconds
    gradient_accumulation_steps: int = 1,
) -> Dict[str, float]:
    """Estimate training time."""
    
    steps_per_epoch = num_samples // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    total_time_seconds = total_steps * time_per_step
    
    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "total_time_seconds": total_time_seconds,
        "total_time_minutes": total_time_seconds / 60,
        "total_time_hours": total_time_seconds / 3600,
        "total_time_days": total_time_seconds / (3600 * 24),
    }


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


class TrainingMetrics:
    """Simple class to track training metrics."""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.metrics: Dict[str, List[float]] = {}
        self.step = 0
    
    def update(self, **kwargs) -> None:
        """Update metrics with new values."""
        
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        self.step += 1
        
        if self.step % self.log_interval == 0:
            self.log()
    
    def log(self) -> None:
        """Log current metrics."""
        
        log_str = f"Step {self.step}"
        
        for key, values in self.metrics.items():
            if values:
                avg = sum(values[-self.log_interval:]) / min(len(values), self.log_interval)
                log_str += f" | {key}: {avg:.4f}"
        
        logger.info(log_str)
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric."""
        
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def save(self, output_path: str) -> None:
        """Save metrics to file."""
        
        with open(output_path, "w") as f:
            json.dump({
                "step": self.step,
                "metrics": {k: {"values": v, "average": sum(v) / len(v) if v else 0} 
                           for k, v in self.metrics.items()}
            }, f, indent=2)


def cleanup_checkpoints(
    output_dir: str,
    keep_last_n: int = 3,
    keep_best: bool = True,
    metric_name: str = "loss",
    lower_is_better: bool = True,
) -> None:
    """Clean up old checkpoints, keeping only the most recent ones."""
    
    checkpoint_dirs = []
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            # Get step number from checkpoint name
            try:
                step = int(item.split("-")[1])
                checkpoint_dirs.append((step, item_path))
            except (IndexError, ValueError):
                continue
    
    # Sort by step
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    
    # Keep last N
    to_keep = set()
    for step, path in checkpoint_dirs[:keep_last_n]:
        to_keep.add(path)
    
    # Keep best if requested
    if keep_best:
        best_metric = float("inf") if lower_is_better else float("-inf")
        best_path = None
        
        for step, path in checkpoint_dirs:
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                if "metrics" in metadata and metric_name in metadata["metrics"]:
                    metric_value = metadata["metrics"][metric_name]
                    
                    if lower_is_better and metric_value < best_metric:
                        best_metric = metric_value
                        best_path = path
                    elif not lower_is_better and metric_value > best_metric:
                        best_metric = metric_value
                        best_path = path
        
        if best_path:
            to_keep.add(best_path)
    
    # Delete old checkpoints
    import shutil
    for step, path in checkpoint_dirs:
        if path not in to_keep:
            logger.info(f"Removing old checkpoint: {path}")
            shutil.rmtree(path)


if __name__ == "__main__":
    # Test utilities
    setup_logging()
    print_gpu_info()
    
    # Test time estimation
    estimate = estimate_training_time(
        num_samples=1_000_000,
        batch_size=4,
        num_epochs=1,
        time_per_step=0.5,
        gradient_accumulation_steps=8,
    )
    print(f"Estimated training time: {format_time(estimate['total_time_seconds'])}")
