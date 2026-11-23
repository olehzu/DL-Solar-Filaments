# Solar Filament Segmentation (COSC 50523 Deep Learning, TCU)

Course project on segmenting solar filaments in full-disk solar images from the Kaggle competition (no longer competing)
"Solar Filament Segmentation Challenge 2026" (MAGFiLO dataset, not included in this repo). We compare a
darkness-threshold baseline and a per-pixel logistic regression (`notebooks/01_logreg_baseline.ipynb`) with a
U-Net (`notebooks/02_unet.ipynb`). `notebooks/03_unet_crops.ipynb` trains the same U-Net on 512 x 512 crops of the
full 2048 x 2048 images and compares it with a U-Net trained on whole images resized to 512.

## Notebook 03: crop-based training at full resolution

Open it in Colab with a GPU runtime. The first cells download the data with the same Kaggle cell as notebooks 01 and 02
and make the same day-based split (same code and seed). Every image and mask is then decoded once into `cache/`, which
takes about 3.5 GB of disk. Training reads one crop at a time from there, so the full-size images never have to sit in
memory. Two U-Nets are trained one after the other, `resized_512` and `crops_2048`. They share the split, loss,
optimizer, augmentation and number of training steps, and both are scored against masks at 2048 x 2048 and at 512 x 512.
We have not timed a full run on Colab yet. Our rough guess is an hour or more on a T4.

Outputs go to `outputs/`, which git ignores: the best weights of each run, `results_<run>.json` with the
configuration and metrics, `comparison_test.csv`, `recall_by_width.csv`, `curves.png` and `qualitative_test.png`.
Colab deletes them when the session ends, so set `OUTPUT_DIR` to a Google Drive folder if you want to keep them. With
`REUSE_WEIGHTS=1` the notebook loads saved weights and only evaluates. The other settings that can be changed through
environment variables are listed at the top of the notebook.

To run it locally you need Python 3.9 or newer with numpy, pillow, matplotlib, pandas, torch (2.3 or newer) and
segmentation-models-pytorch. We checked it with Python 3.13, torch 2.15 and segmentation-models-pytorch 0.6.0. On macOS
and Windows the DataLoaders run in the main process (`NUM_WORKERS=0`), because Jupyter there cannot pass classes
defined in a notebook to worker processes.

## Planned work

- Full-resolution / crop-based training (notebook 03, waiting on a full Colab run)
- A second architecture
- A hyperparameter comparison
- A disc mask for the darkness-threshold baseline
