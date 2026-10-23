"""
Anti-Money Laundering (AML) End-to-End Pipeline & Live Demo Runner
Automates the full cycle: SAML-D Download & Load -> DW ETL -> OLAP Operations -> 4 ML Mining Models -> Dashboard Launch.
"""

import sys
import os
import subprocess
import time

sys.path.insert(0, os.path.abspath("."))

def print_header(title):
    print("\n" + "=" * 70)
    print(f"🛡️  {title.upper()}")
    print("=" * 70)

def run_step(step_name, command):
    print_header(step_name)
    print(f"Executing: {command} ...")
    start = time.time()
    result = subprocess.run(command, shell=True, check=True)
    duration = time.time() - start
    print(f"✅ Finished {step_name} in {duration:.2f}s\n")

def main():
    print("""
    ==================================================================
       ANTI-MONEY LAUNDERING (AML) DATA WAREHOUSE & DATA MINING DEMO
       Course Topic: Anti-Money Laundering (Team: Code Hunters)
    ==================================================================
    1. Dataset Loading (SAML-D from Kaggle) & Description
    2. Data Warehouse (DW) Star Schema & ETL Pipeline
    3. Sample OLAP Operations (Roll-up, Drill-down, Slice, Dice, Pivot)
    4. Classification (Suspicious / Non-Suspicious SAR Detection)
    5. Clustering (Account Behaviour Groups, 7 algorithms)
    6. Association Rules (Suspicious Sequence & Typology Rules)
    7. Prediction (Calibrated AML Risk Score 0-100)
    8. Interactive Compliance Dashboard & Visualizations
    ==================================================================
    """)
    
    venv_python = os.path.abspath(".venv/bin/python")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable
    
    # 1. Download (first run only) and load the SAML-D dataset
    run_step("Step 1: SAML-D Dataset Loading & Account Profiling", f"{python_bin} data/saml_loader.py")
    
    # 2. Run DW ETL Pipeline
    run_step("Step 2: Data Warehouse ETL Pipeline (Star Schema)", f"{python_bin} warehouse/etl_pipeline.py")
    
    # 3. Test OLAP Engine
    run_step("Step 3: Sample OLAP Operations Demonstration", f"{python_bin} warehouse/olap_engine.py")
    
    # 4. Train All Data Mining Models
    run_step("Step 4: Machine Learning & Data Mining Suite", f"{python_bin} models/train_all.py")
    
    print_header("Step 5: Live Interactive Dashboard Ready")
    print("To launch the modern web dashboard, run:")
    print("👉  streamlit run dashboard/app.py")
    print("\nDemo pipeline executed successfully!")

if __name__ == "__main__":
    main()
