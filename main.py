import subprocess


def run_step(script_name):
    print(f"\n>>> Executing Stage: {script_name}...")
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        print(f"!! Error in {script_name}. Pipeline halted.")
        raise SystemExit(1)


def main():
    print("--- STARTING GLAUCOMA DETECTION PIPELINE ---")

    # Step 1: Data Preparation for ACRIMA
    run_step("prepare_acrima_metadata.py")

    # Step 2: Baseline model training
    run_step("train_baseline.py")

    # Step 3: GAN Training (Synthetic Data Generation)
    run_step("train_gan.py")

    # Step 4: Incremental Learning Integration
    run_step("train_incremental.py")

    # Step 5: Performance Evaluation
    run_step("evaluate.py")

    print("\n[SUCCESS] Stage-I Implementation Complete.")


if __name__ == "__main__":
    main()
