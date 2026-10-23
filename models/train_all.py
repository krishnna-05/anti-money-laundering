"""
Master Runner: Executes and Trains all Data Mining & Machine Learning Modules
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from models.classification_module import train_classifier
from models.clustering_module import train_behavioral_clustering
from models.association_module import mine_association_rules
from models.prediction_module import train_risk_predictor

def train_all_models():
    print("=" * 70)
    print("🚀 ANTI-MONEY LAUNDERING (AML) MACHINE LEARNING & MINING SUITE")
    print("=" * 70)
    
    print("\n[STEP 1/4] Running Classification Model...")
    train_classifier()
    
    print("\n[STEP 2/4] Running Behavioral Clustering Model...")
    train_behavioral_clustering()
    
    print("\n[STEP 3/4] Running Association Rule Mining Engine...")
    mine_association_rules()
    
    print("\n[STEP 4/4] Running Continuous Risk Score Predictor...")
    train_risk_predictor()
    
    print("\n" + "=" * 70)
    print("✅ ALL 4 AML DATA MINING MODULES SUCCESSFULLY TRAINED AND SAVED!")
    print("=" * 70)

if __name__ == "__main__":
    train_all_models()
