
import os
import deepspeed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_deepspeed():
    ds_path = os.path.dirname(deepspeed.__file__)
    target_file = os.path.join(ds_path, "runtime", "zero", "stage_1_and_2.py")
    
    logger.info(f"Patching DeepSpeed at: {target_file}")
    
    with open(target_file, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    patched = False
    for line in lines:
        if "from deepspeed.runtime.zero.muon.original_muon import muon_update" in line:
            logger.info("Found offending line. Commenting it out.")
            new_lines.append(f"# PATCHED BY ZIZOGPT: {line}")
            patched = True
        else:
            new_lines.append(line)
            
    if patched:
        with open(target_file, "w") as f:
            f.writelines(new_lines)
        logger.info("DeepSpeed successfully patched!")
    else:
        logger.info("DeepSpeed was already patched or line not found.")

if __name__ == "__main__":
    patch_deepspeed()
