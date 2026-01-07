
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
    
    # Target the source of the crash: original_muon.py
    relative_path = os.path.join("deepspeed", "runtime", "zero", "muon", "original_muon.py")
    
    for sp in site_packages:
        possible_path = os.path.join(sp, relative_path)
        if os.path.exists(possible_path):
            target_file = possible_path
            break
            
    if not target_file:
        # Fallback search
        logger.warning("Standard site-packages search failed. Searching /usr/local/lib...")
        files = glob.glob(f"/usr/local/lib/python*/dist-packages/{relative_path}")
        if files:
            target_file = files[0]
            
    if not target_file:
        logger.error(f"Could not locate {relative_path}")
        return

    logger.info(f"Patching DeepSpeed at: {target_file}")
    
    with open(target_file, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    patched = False
    for line in lines:
        # Comment out the compiler decorator
        if "@compiler.compile()" in line:
            logger.info("Found offending decorator. Commenting it out.")
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
