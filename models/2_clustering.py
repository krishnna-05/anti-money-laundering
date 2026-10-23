"""
Rubric entry point: Clustering. The implementation lives in models/clustering_module.py.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from models.clustering_module import train_behavioral_clustering

if __name__ == "__main__":
    train_behavioral_clustering()
