# Run instructions

## 1. Create the environment

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate dip-efficient-nlm
```

or pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.11 was used for the submitted run.

## 2. Fast correctness/demo path

Open and run `Efficient_NLM_Demo.ipynb`. It compares direct and accelerated NLM on a controlled crop, verifies numerical equivalence, and explains the complexity reduction.

## 3. Reproduce the complete native-image experiment

From the package root:

```bash
MPLCONFIGDIR=/tmp/dip_mpl_cache python code/best4_final_experiment.py
```

The package-relative script reads the four confirmation images from `data/`, the separate gate-training images from `gate_training_data/`, and fixed parameter files from `parameters/`. It recreates CSVs, processed images, and report figures under `results_reproduced/`.

The full D5 confirmation is deliberately compute-heavy and may take several minutes. The exact NLM equivalence check is much faster.

## 4. Protocol and timing interpretation

- Gaussian noise uses `σ∈{10,25,50}`, clipping to `[0,255]`.
- Confirmation seed: `20261029`; gate-tuning seed: `20260827`.
- Native comparison: `K=7`, `S=21`; exact `h/σ=0.6` was frozen before confirmation.
- Timers surround complete public method calls; metrics and file I/O are excluded.
- Fusion timing includes expert construction, scoring, PCA, KD-tree retrieval, reranking, and reconstruction.
- Runtime is hardware-dependent; compare methods within the same run.
- Direct NLM is benchmarked on fixed 96×96 crops to keep literal `O(N S² K²)` verification practical.

## 5. Rebuild the report

The rendered PDF and editable DOCX are already included. `code/build_report.py` reads all numbers from `results/`; it requires `python-docx` and a local Chromium/Chrome executable for PDF rendering. `report/REPORT_SOURCE.md` is the readable report source.
