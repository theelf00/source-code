import os
import subprocess

def run_step(script_name):
    print(f"\n>>> Executing Stage: {script_name}...")
    result = subprocess.run(['python', script_name])
    if result.returncode != 0:
        print(f"!! Error in {script_name}. Pipeline halted.")
        exit(1)

def main():
    print("--- STARTING GLAUCOMA DETECTION PIPELINE ---")
    
    # Step 1: Data Preparation for ACRIMA
    run_step('prepare_acrima_metadata.py')
    
    # Step 2: GAN Training (Synthetic Data Generation)
    # This addresses the scarcity of labeled medical data [cite: 51]
    run_step('train_gan.py')
    
    # Step 3: Incremental Learning Integration
    # Adapts to new data while preserving previously learned knowledge [cite: 54]
    run_step('train_incremental.py')
    
    # Step 4: Performance Evaluation
    # Calculates Accuracy, Precision, Recall, and F1-score 
    run_step('evaluate.py')

    print("\n[SUCCESS] Stage-I Implementation Complete.")

if __name__ == "__main__":
    main()