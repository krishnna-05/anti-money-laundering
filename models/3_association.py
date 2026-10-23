"""
Rubric entry point: Association Rules. The implementation lives in models/association_module.py.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from models.association_module import mine_association_rules

if __name__ == "__main__":
    mine_association_rules()
