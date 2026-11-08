---
name: python-notebook
description: Enforce ML/NLP notebook standards for Python/Colab. Mounts Drive in cell 1, scopes variables per cell, Vietnamese diacritic comments on key logic only, zero icons, pre/post pipeline verification, auxiliary scripts stored in Scripts/.
---

# Python Notebook Engineering Skill

Standardized workflow and constraints for creating, refactoring, and maintaining Jupyter and Google Colab notebooks for machine learning and NLP pipelines.

## 1. Core Constraints

### 1.1. Zero Icons / Emojis
- Strictly forbid any icons, emojis, or pictorial symbols across the entire notebook:
  - Code strings, `print()` outputs, progress displays, and logging.
  - Markdown headings, callout boxes, and text descriptions.
  - Code comments (`# ...`).
  - Matplotlib / Seaborn / Plotly titles, axis labels, legends (`plt.title`, `plt.xlabel`, `plt.ylabel`).
- Use plain text and technical indicators only.

### 1.2. Selective Comments in Vietnamese with Diacritics
- Only comment on non-trivial logic: algorithmic decisions, complex data manipulations, tensor transformations, custom loss formulations, or error-handling mechanisms.
- Do NOT comment on self-explanatory code (e.g., standard imports, reading standard CSV files, simple variable assignments).
- Language requirement: All comments MUST be written in grammatically correct Vietnamese with full diacritics (tiếng Việt có dấu).

### 1.3. Per-Cell Variable Scoping
- Do not pool all variables or configurations into a single large global configuration cell.
- Define cell-specific hyperparameters, flags, thresholds, and input paths at the very top of the respective cell executing that task.
- Ensure cells remain modular, self-contained, and easy to run iteratively.

### 1.4. Google Drive Persistence in Cell 1
- The first code cell must mount Google Drive (`google.colab.drive`) and configure persistent project paths.
- All critical outputs (trained weights, model checkpoints, preprocessed data, metrics logs) must point to these Drive directories to avoid data loss on runtime disconnection.

### 1.5. Organization of Auxiliary and Generated Scripts
- Any standalone, auxiliary, helper, conversion, or redundant script files (`.py`, `.sh`, etc.) generated during pipeline development or data processing MUST be stored inside the `Scripts/` directory.
- Never leave generated or temporary script files scattered in the project root directory.

---

## 2. Pipeline Execution Workflow

### Phase 1: Pre-flight Verification (Before Writing Code)
- Inspect the overall project pipeline and data flow.
- Verify the existence, schema, and paths of input files (e.g., train/dev/test datasets, external teencode dictionaries).
- Check upstream and downstream requirements for the current notebook to ensure seamless integration.
- Determine the exact list of target artifacts required (checkpoints, evaluation reports, predictions).

### Phase 2: First Cell Setup (Drive & Directories)
The initial code cell must establish Google Drive connection and working directories:

```python
from google.colab import drive
import os

drive.mount('/content/drive')

project_dir = '/content/drive/MyDrive/NLP_PhoBert_ViHSD'
checkpoints_dir = os.path.join(project_dir, 'checkpoints')
outputs_dir = os.path.join(project_dir, 'outputs')

os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)
print(f'Thu muc lam viec: {project_dir}')
```

### Phase 3: Cell-by-Cell Construction
- **Top of Cell:** Declare variables, thresholds, and configurations needed for this specific step.
- **Body:** Implement lean, modular Python code.
- **Comments:** Insert concise Vietnamese comments with diacritics only at complex logic boundaries.

Example cell structure:

```python
# Cau hinh tham so cho qua trinh xu ly van ban
max_sequence_length = 256
batch_size = 32
teencode_path = 'data/teencode.json'

# Chuan hoa tu ngu teencode dua tren tu dien anh xa truoc khi tach tu phuc
def normalize_teencode(text, mapping_dict):
    words = text.split()
    return ' '.join([mapping_dict.get(w, w) for w in words])
```

### Phase 4: Post-flight Verification (After Writing Code)
- Trace the execution order to ensure no hidden state or out-of-order execution dependencies exist.
- Verify all generated artifacts, model checkpoints, and evaluation tables are directed to persistent Drive storage.
- Check compliance with the pipeline requirements defined in Phase 1.
- Audit the final notebook file to confirm zero icons or emojis were introduced.
