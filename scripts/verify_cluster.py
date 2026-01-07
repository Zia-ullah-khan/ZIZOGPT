#!/usr/bin/env python
"""
ZIZOGPT Cluster Verification Script
Verifies GPU availability, P2P communication, and basic tensor operations.
"""

import os
import torch
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_cluster():
    logger.info("Starting Cluster Verification...")
    
    # 1. Check GPU Availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        return False
    
    device_count = torch.cuda.device_count()
    logger.info(f"Detected {device_count} GPUs.")
    
    if device_count < 4:
        logger.warning(f"Expected 4 GPUs, found {device_count}. Proceeding with caution.")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")

    # 2. Check P2P Communication
    logger.info("Checking Peer-to-Peer (P2P) Access...")
    for i in range(device_count):
        for j in range(device_count):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                status = "OK" if can_access else "FAIL"
                log_level = logging.INFO if can_access else logging.WARNING
                logger.log(log_level, f"P2P {i} -> {j}: {status}")

    # 3. Functional Test (Forward/Backward Pass)
    logger.info("Running Functional Test (Forward/Backward Pass)...")
    try:
        # Create a simple matrix multiplication on each GPU to warm up and test
        for i in range(device_count):
            logger.info(f"Testing GPU {i}...")
            device = torch.device(f"cuda:{i}")
            
            # Create large tensors to test memory and compute
            a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
            b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
            target = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
            
            # Forward
            start_time = time.time()
            c = torch.matmul(a, b)
            
            # Backward (simulate)
            loss = torch.nn.functional.mse_loss(c, target)
            loss.backward()
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            logger.info(f"GPU {i} Test Passed! Time: {elapsed:.4f}s")
            
    except Exception as e:
        logger.error(f"Functional Test Failed: {e}")
        return False

    logger.info("Cluster Verification Complete. System Ready.")
    return True

if __name__ == "__main__":
    verify_cluster()
