# Solar Filament Segmentation (COSC 50523 Deep Learning, TCU)

Course project for COSC 50523 Deep Learning at TCU (midterm progress stage). The task is binary semantic
segmentation of solar filaments in full-disk solar images from the MAGFiLO dataset, as posed by the Kaggle
competition "Solar Filament Segmentation Challenge 2026". Ground-truth masks are drawn from COCO-style polygon
annotations. We compare a non-learning darkness-threshold baseline and a per-pixel logistic regression
(NumPy, at 256 x 256) with a U-Net with a ResNet34 encoder (PyTorch, `segmentation_models_pytorch`, at 512 x 512).
All models use the same image-level, day-based train/val/test split and the same pooled pixel metrics.

## Data (not included)

- Competition: **Solar Filament Segmentation Challenge 2026**, Kaggle slug `filament-segmentation-2026`. Dataset: MAGFiLO.
- Per the competition rules, the data is licensed for **non-commercial / academic use only**. It is not, and must not be, committed to this repository.
- Format: one COCO-style annotation JSON (`images`, `annotations` with polygon `segmentation`, 4 `categories`) plus 2048 x 2048 JPEGs in `train_images/`, named by timestamp (e.g. `20140609195854Bh.jpeg`; the first 8 characters are YYYYMMDD).
- To get it:
  1. On Kaggle, open the competition page, click **Join Competition** and accept the rules. Without this the download returns **403**.
  2. Run the "Get the data (Colab only)" cell in either notebook. It asks for your Kaggle API token via `getpass` at runtime (or lets you upload a legacy `kaggle.json`), downloads, and unzips into `DATA_DIR` (default `/content/filament-segmentation-2026`). The token is never stored in the notebook. Never paste it into a cell, and never commit `kaggle.json`.
  3. Locally (outside Colab), download it yourself with the Kaggle CLI (`kaggle competitions download -c filament-segmentation-2026`), unzip, and set the environment variable `DATA_DIR` to the unzipped folder. The notebooks find the annotation file by searching `DATA_DIR` for `*Annot*train*.json`.

## Dataset statistics

From the Colab run of `01_logreg_baseline.ipynb` (2026-10-21, masks at 256 x 256, seed 42, `CATEGORY_IDS = None`):

- 1154 image records and 8199 annotations in the training JSON. All 1154 images were found and loaded, and none has an empty mask.
- Categories (supercategory `filament`): `1 Left` (2535 annotations), `2 Right` (2590), `3 Unidentifiable` (3074), `4 Ambiguous` (0).
- Filament pixel fraction over all images: 0.6150% at 256 x 256 (0.4680% at 512 x 512 in notebook 02).
- 296 file names appear in more than one image record (447 extra records), so the 1154 records refer to 707 distinct
  image files. Records with the same file name have the same day, so they always land in the same split.

| Split | Images | Days | First day | Last day | Filament pixel fraction | Annotations |
|---|---|---|---|---|---|---|
| train | 809 | 452 | 2011-01-09 | 2022-08-03 | 0.6026% | 5691 |
| val | 172 | 101 | 2011-01-28 | 2022-07-21 | 0.7298% | 1344 |
| test | 173 | 103 | 2011-01-14 | 2022-06-17 | 0.5587% | 1164 |

"Images" counts image records, including the repeated file names above. The per-split annotations sum to 8199,
matching the JSON. Notebook 02 produces the identical split (same image counts, days and date ranges). At its
512 x 512 mask resolution the filament pixel fractions are train 0.4590%, val 0.5565% and test 0.4223%.

Leakage checks (asserted in both notebooks): no day appears in more than one split, and no file name appears in more
than one split. Both checks passed.

## Setup

**Google Colab (recommended).** Open the notebook from GitHub in Colab (File > Open notebook > GitHub). For
`02_unet.ipynb` select a GPU (Runtime > Change runtime type > T4 GPU). Notebook 02 installs `segmentation-models-pytorch`
itself, and the download cell installs `kaggle`.

**Local.**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Without a GPU, notebook 02 runs but is very slow at the default settings. On Windows, the U-Net DataLoaders
(`num_workers=2`) may fail inside Jupyter because worker processes cannot import classes defined in a notebook.
If that happens, set `num_workers=0` in your local copy.

## How to run

Run the notebooks in this order. Each one is self-contained: it loads the data, draws the masks, and makes the split itself.

| Notebook | What it does / produces | Approx. runtime |
|---|---|---|
| `notebooks/01_logreg_baseline.ipynb` | Dataset statistics and leakage checks, darkness-threshold baseline, per-pixel logistic regression (HW2-style), val-tuned thresholds, test metrics, prediction plots. Writes `outputs/results_baselines.json` (git-ignored). | Not measured |
| `notebooks/02_unet.ipynb` | Same statistics and leakage checks, U-Net (ResNet34) training with BCE + Dice loss, best epoch by validation Dice, val-tuned threshold, test metrics, comparison table, prediction plots. Reads `outputs/results_baselines.json` if present, otherwise uses the baseline numbers hardcoded from notebook 01. The best weights are kept in memory only; no checkpoint is saved. | Training: about 7.7 min for 15 epochs (29 to 34 s per epoch, including validation) on a Colab GPU (the notebook's saved runtime setting is T4). Full notebook including data download and loading: not measured |

Environment variables read by the notebooks: `DATA_DIR` (both), `OUTPUT_DIR` (both, default `outputs`), and
`EPOCHS`, `BATCH`, `ENCODER_WEIGHTS` (notebook 02; set `ENCODER_WEIGHTS=None` to train from scratch).

**Smoke test (no real data needed).** This generates a tiny synthetic COCO dataset in a temp directory and executes both
notebooks on CPU (1 epoch, no pretrained weights):

```bash
pip install -r requirements.txt -r requirements-dev.txt
python tests/make_synthetic_coco.py      # optional: just generate the data and print its DATA_DIR
pytest tests/ -v
```

## Results (preliminary)

| Model | Resolution | Split | IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|
| Darkness-threshold baseline | 256 x 256 | test | 0.0039 | 0.0078 | 0.0040 | 0.4082 |
| Per-pixel logistic regression (9 x 9 patch) | 256 x 256 | test | 0.0138 | 0.0271 | 0.0139 | 0.5848 |
| U-Net, ResNet34 encoder (ImageNet init) | 512 x 512 | test | 0.5331 | 0.6955 | 0.6824 | 0.7090 |

Internal held-out **test** split (15% of images, split by day), single run, seed 42. The probability threshold
(and darkness cutoff for the baseline) is tuned on the validation split. Metrics are pooled over all pixels of
the test images. These are **not** Kaggle leaderboard scores. The rows use different resolutions (256 vs 512).
All three rows were reproduced exactly in fresh Colab runs of both notebooks on 2026-10-21 (for the U-Net, the
per-epoch validation Dice log was also identical). The U-Net was selected at epoch 14 (validation Dice 0.6945 at
threshold 0.5), and the tuned threshold was 0.65. For the logistic regression the tuned probability threshold was
0.95, and for the baseline the darkness cutoff was 0.60.

This supports our choice of a deep neural network. It gave us ideas for the future
development, including trying alternative segmentation methods, and altering hyperparameters.

## Attribution

| Component | Origin |
|---|---|
| Data loading, polygon-to-mask drawing, day-based split, statistics and leakage checks, patch features, pixel sampling, metrics, threshold tuning, U-Net training loop and loss, test scripts | Our own code (MIT, see `LICENSE`) |
| `sigmoid`, `initialize_with_zeros`, `propagate`, `optimize` in notebook 01 | Adapted from our HW2 course assignment (COSC 50523) |
| U-Net architecture and encoders | [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) (third-party library) |
| ImageNet-pretrained ResNet34 encoder weights | Downloaded at runtime by `segmentation_models_pytorch` (not redistributed here) |
| NumPy, Pillow, Matplotlib, pandas, PyTorch, Kaggle CLI | Third-party libraries |
| MAGFiLO data | Kaggle competition `filament-segmentation-2026` (not redistributed here; non-commercial / academic use per competition rules) |

## License

The team's own code is released under the MIT License (see [`LICENSE`](LICENSE)), Copyright (c) 2026 Turner DeMott,
Oleh Zubariev, Balaji Kannan. The license does not cover the HW2-derived functions, the MAGFiLO data, or third-party
libraries and pretrained weights, which keep their own terms.

## What is not in this repo

- The dataset (images, annotation JSON, archives). Download it from Kaggle as described above.
- Model weights or checkpoints, cached arrays, and run outputs (`outputs/` is git-ignored).
- Kaggle tokens or `kaggle.json`, or any other credential.

## AI assistance

An AI agent was used for gathering information, exploring design ideas, as well as implementation and debugging of certain areas of code and documentation. The team members were responsible for designing, training, and evaluating models based on the performance parameters. The team members are responsible for any written analysis and conclusions, as well as the final decisions on the model modifications. All AI-generated content was reviewed and verified by the team. The team members can explain the thought process behind the design and each area of the written code, if asked.

## Planned work

- Full-resolution / crop-based training (requested by the professor)
- A second architecture (requested by the professor)
- A hyperparameter comparison (requested by the professor)
- A disc mask for the darkness-threshold baseline

## References

- A. Ahmadzadeh, D. J. Kempton, Q. Li, and A. A. Pevtsov, "Solar Filament Segmentation Challenge 2026," Kaggle, 2026. https://www.kaggle.com/competitions/filament-segmentation-2026
- G. Zhu, G. Lin, X. Yang, and C. Zeng, "Flat U-Net: An Efficient Ultralightweight Model for Solar Filament Segmentation in Full-disk Hα Images," ApJ 980, 176, 2025. https://doi.org/10.48550/arXiv.2502.07259
- O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015.
- K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," 2016.
- D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," 2015.
- F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation," 2016 (Dice loss).
- P. Iakubovskii, "Segmentation Models Pytorch," software. https://github.com/qubvel/segmentation_models.pytorch
