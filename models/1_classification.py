"""
Rubric entry point: Classification. The implementation lives in models/classification_module.py.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from models.classification_module import train_classifier

if __name__ == "__main__":
    train_classifier()
