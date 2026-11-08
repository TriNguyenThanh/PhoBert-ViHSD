# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository focuses on research and model enhancement for Vietnamese Hate Speech Detection (ViHSD), based on the benchmark paper *"A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts"* (Luu et al., 2021).

The task is a 3-class classification problem on Vietnamese social media comments:
- `0`: CLEAN (~83% of dataset)
- `1`: OFFENSIVE (~7% of dataset)
- `2`: HATE (~10% of dataset)

Primary evaluation metric is **Macro-F1** due to severe class imbalance.

## Repository Structure

- `data/`: Contains dataset files (`train.csv`, `dev.csv`, `test.csv` with columns `free_text`, `label_id`) and normalization resources (`teencode.json`, `teencode.xlsx`, `teencode.csv`).
- `Paper/`: Reference publication (`2021_Luu_ViHSD_A_Large_Scale_Dataset.pdf`) and original baseline notebook (`ViHSD_monolingual_Transformers.ipynb`).
- `Notebook/`: Jupyter notebooks for experiments and modeling.
- `Pipeline/`: Modular NLP pipelines (preprocessing, feature extraction, training, evaluation).
- `Scripts/`: Auxiliary scripts (data merging, text normalization, dataset utilities).
- `Docs/`: Research documentation, experiments tracking, and notes.

## Environment & Dependency Management

Always use `uv` and `requirements.txt` to manage virtual environments and packages:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Run Python scripts with uv
uv run python Scripts/<script_name>.py
```

## Coding & Research Standards

1. **Academic Tone & Style:**
   - Maintain clear, precise, and concise academic writing.
   - Absolutely no decorative emojis or icons in source code, notebooks, or commit messages.
   - Use standard Vietnamese with full diacritics for key explanatory comments in notebooks and scripts.

2. **Jupyter Notebook Guidelines:**
   - **File naming convention:** Notebook files must strictly follow the format `checkpoint01_<name>.ipynb`, `checkpoint02_<name>.ipynb`, etc.
   - Cell 1 must handle Google Drive mounting or environment path resolution cleanly.
   - Declare hyperparameters and input variables at the top of relevant cells.
   - Always run pre-pipeline and post-pipeline data verification checks (assert shapes, check nulls).
   - Place non-trivial helper logic and heavy pre-processing routines in standalone scripts within `Scripts/`.

3. **Domain-Specific NLP Rules:**
   - **Do not strip stopwords indiscriminately:** Personal pronouns (`mày`, `tao`, `nó`, `thằng`, `con`) and negation terms (`không`, `chẳng`, `chả`) carry essential semantics for detecting hate speech targets and stance.
   - **Preserve emotional polarity:** Do not strip emojis blindly; convert or retain them for sentiment and offensive polarity features.
   - **Handle Class Imbalance:** Cross-entropy loss without class weighting is insufficient. Always evaluate with Macro-F1 and utilize Weighted Cross-Entropy Loss, Focal Loss, or balanced sampling.

4. **Git Commit Guidelines:**
   - **Commit Message Format:** Always strictly use the format `<action>(<feature>): <message>` (e.g., `feat(preprocessing): add teencode normalization`, `fix(pipeline): handle empty string edge case`).
   - **No Co-Author Attribution:** Absolutely do not append `Co-Authored-By` or any co-author lines to commit messages.

5. **Baseline Compatibility & Benchmarking Rules (`Paper/ViHSD_monolingual_Transformers.ipynb`):**
   - **Data Flow & Variable Naming:** Maintain structural alignment with the original baseline notebook:
     - Dataset loading from `data/train.csv`, `data/dev.csv`, `data/test.csv` using columns `free_text` and `label_id`.
     - Consistent data variable conventions: `train_X, train_y`, `dev_X, dev_y`, `test_X, test_y`.
     - PyTorch Dataset structure following `BuildDataset(encodings, labels)` returning dictionary with tensor features and `'labels'`.
   - **Metric & Evaluation Alignment:**
     - Preserve standard evaluation logic: `y_pred = np.argmax(predictions, axis=-1)` and `y_true = test_y`.
     - Mandatory evaluation metrics: Macro-F1 (primary metric via `average='macro'`), Micro-F1 (`average='micro'`), per-class F1, and `confusion_matrix(y_true, y_pred)`.
     - Class label order: Always preserve the exact class order `0: clean`, `1: offensive`, `2: hate` across all metrics, classification reports, and confusion matrix visualizations.
   - **Reproducibility & Fair Comparison:**
     - Fix random seed (`seed=42`) across Python, NumPy, PyTorch, and Hugging Face Trainer to ensure fair and deterministic comparison against the baseline.
     - Isolate variables strictly per checkpoint as defined in `Docs/roadmap.md` so that ablation gains are directly attributable and reproducible.
