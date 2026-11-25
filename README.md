# tem-classifier

Particle-species classifier for cryo-EM particle images. A small transformer
model (`CryoFormerLite`) is trained on labeled particle stacks, then used to
predict labels and class probabilities for new particles, supplied either as a
CSV or pulled directly from a [CryoSPARC](https://cryosparc.com) job.
Predictions can be fed back in to build a self-labeled training set.

## Getting started

### 1. Set up the Python environment

Python 3.12 or newer is required. Create an environment and install the
pinned dependencies from `requirements.txt`:

```bash
conda create -n protein-mixtures python=3.12
conda activate protein-mixtures
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for training; the code falls back to CPU
automatically. On Linux with a specific CUDA version, you may need to install `torch` and
`torchvision` from the matching index at [pytorch.org](https://pytorch.org/get-started/locally/).

### 2. Prepare your data

- **Training CSV** with one row per particle. Required columns: `blob/path`
  (path to the `.mrc` stack, relative to `--base_dir`), `blob/idx` (index of
  the particle in the stack) and `particle` (species name).
- **Label map** (JSON) mapping species name to class index, e.g.
  `{"ApoF": 0, "Beta-gal": 1}`. Saved as `particle2label.json` next to the
  model for inference.

## Running

All scripts are CLI tools; use `--help` for the full option list.

**Train** (add `--check` first to smoke-test a single batch):

```bash
python cryoclassifier.py --input_csv training.csv --label_map particle2label.json \
    --base_dir /path/to/data/ --output_dir models/exp1/
```

The best checkpoint (`cryo_model_<val_acc>.pth`) is written to `--output_dir`.
Training curves are logged to `runs/`; view them with `tensorboard --logdir runs`.

**Predict** with a trained model, from a CSV or a CryoSPARC job:

```bash
python inference_cryoclassifier.py --model_dir models/exp1/ --input_csv particles.csv
python inference_cryoclassifier.py --model_dir models/exp1/ --job_id J123 --project_id P4
```

**Build a self-labeled training set** from the predictions (semi-supervised round):

```bash
python update_training_set.py --training_path training.csv \
    --predictions_path predictions.csv --probs_path predictions.probs.npy \
    --label_map particle2label.json
```

`VizEmbeddings.ipynb` is a notebook for visualizing learned embeddings.

## Configuration

**Command-line options** (key ones; defaults in each script's `--help`):

| Option | Scripts | Meaning |
|---|---|---|
| `--label_map` / `--num_classes` | train, inference | Class definitions (the map takes precedence) |
| `--base_dir` | train, inference | Root directory for relative particle paths (training default: `/mnt/data/CS-protein-mixture/`) |
| `--image_size` | train, inference | Particle crop size in pixels (default 128) |
| `--pixel_size` | train, inference | Pixel size in Å (default 2.053125); must match between training and inference |
| `--epochs`, `--use_focal_loss` | train | Training length; focal loss for class imbalance |
| `--quantile`, `--seed` | seeding | Confidence cutoff (default 0.75) and RNG seed |

**CryoSPARC credentials** (only for `--job_id` inference) are read from a
`.env` file in the working directory:

```
CRYOSPARC_LICENSE_ID=...
CRYOSPARC_EMAIL=...
CRYOSPARC_PASSWORD=...
CRYOSPARC_HOST=...
CRYOSPARC_PORT=...
```

Keep `.env` out of version control.

**Hyperparameters** (learning rate 3e-4, weight decay, label smoothing,
gradient clipping) are set in code in `cryoclassifier.py` and `training.py`.

## Repository layout

| File | Purpose |
|---|---|
| `cryoclassifier.py` | Training CLI |
| `inference_cryoclassifier.py` | Inference CLI |
| `update_training_set.py` | Seeded training-set builder |
| `models.py`, `losses.py`, `training.py` | Model, losses, optimizer/epoch loops |
| `dataset.py`, `label_utils.py`, `utils_cs.py` | Data loading, label maps, CryoSPARC helpers |
