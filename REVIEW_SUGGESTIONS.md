# Project Analysis & Suggestions

## Current State Snapshot
- The project has a complete end-to-end glaucoma detection pipeline: metadata preparation, GAN training, incremental classifier training, evaluation, and a Flask inference app.
- Core functionality is present, but the codebase is currently research-prototype style rather than production-ready.
- Key risks are around reproducibility, model loading assumptions, security hardening in the web app, and missing operational documentation.

## Key Findings

### 1) Documentation is too minimal
- `readme.md` only contains a dataset link and does not explain setup, training, evaluation, or serving workflow.
- This creates onboarding friction and makes reproducibility difficult.

### 2) Inference app robustness/security gaps
- `app.py` assumes uploaded file exists and is valid; there is no file-type validation, size limit, or graceful error handling for corrupt images.
- The app runs with `debug=True`, which should not be used in deployment.
- In-memory history is useful for demos but not durable.

### 3) Training/evaluation reproducibility is limited
- There are no explicit random seeds for `torch`, `numpy`, and Python random.
- Dataset split strategy is not explicit; evaluation appears to run on the same metadata source used for training, which can overestimate model quality.

### 4) Model lifecycle concerns
- The classifier uses `resnet18(pretrained=True)`, which is deprecated in modern torchvision in favor of explicit `weights` APIs.
- `train_incremental.py` initializes `old_model` but does not load a saved baseline checkpoint, reducing the intended effect of knowledge distillation.

### 5) Data pipeline safety checks are missing
- Dataset loading relies on positional CSV columns (`iloc`) and does not validate expected headers or file existence before opening images.
- Transform normalization differs between GAN path and classifier path (may be intentional, but should be documented).

## Prioritized Suggestions

### High Priority (stability and correctness)
1. Expand README with:
   - Environment setup
   - Data preparation
   - Training/evaluation commands
   - Flask serving instructions
   - Directory structure and expected artifacts
2. Add train/validation/test split logic and evaluate on holdout test set only.
3. Add deterministic seeding utility used by all training/evaluation scripts.
4. Load an actual previous checkpoint in incremental learning before distillation.
5. Replace `debug=True` with environment-driven configuration in Flask app.

### Medium Priority (maintainability)
1. Migrate to torchvision `weights` argument for ResNet18.
2. Add config management (YAML/TOML or argparse flags) instead of hard-coded constants.
3. Add input validation and structured error responses in `app.py`.
4. Add small unit tests for dataset parsing and loss function behavior.

### Low Priority (quality-of-life)
1. Add model/version metadata with checkpoint saves (epoch, metrics, config hash).
2. Add logging abstraction instead of `print` statements.
3. Add pre-commit formatting/linting (e.g., black, ruff).

## Suggested Immediate Next Sprint
1. Documentation pass + reproducible runbook.
2. Proper dataset split and updated evaluation script.
3. Incremental learning checkpoint flow fix.
4. Flask hardening for upload validation and production mode toggle.
