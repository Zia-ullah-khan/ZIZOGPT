#!/usr/bin/env python
"""
Quick-start script for SFT training ZIZOGPT
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.sft import main

if __name__ == "__main__":
    main()
