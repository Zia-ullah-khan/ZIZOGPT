
import os
import site
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_deepspeed():
    # Find site-packages
    site_packages = site.getsitepackages()
    target_file = None
    
    for sp in site_packages:
        possible_path = os.path.join(sp, "deepspeed", "runtime", "zero", "stage_1_and_2.py")
        if os.path.exists(possible_path):
            target_file = possible_path
            break
            
    if not target_file:
        # Fallback search
        logger.warning("Standard site-packages search failed. Searching /usr/local/lib...")
        files = glob.glob("/usr/local/lib/python*/dist-packages/deepspeed/runtime/zero/stage_1_and_2.py")
        if files:
            target_file = files[0]
            
    if not target_file:
        logger.error("Could not locate deepspeed/runtime/zero/stage_1_and_2.py")
        return

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
