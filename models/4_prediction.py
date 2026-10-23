"""
Rubric entry point: Prediction. The implementation lives in models/prediction_module.py.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from models.prediction_module import train_risk_predictor

if __name__ == "__main__":
    train_risk_predictor()
