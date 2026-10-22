# Solar Filament Segmentation (COSC 50523 Deep Learning, TCU)

Course project on segmenting solar filaments in full-disk solar images from the Kaggle competition
"Solar Filament Segmentation Challenge 2026" (MAGFiLO dataset, not included in this repo). We compare a
darkness-threshold baseline and a per-pixel logistic regression (`notebooks/01_logreg_baseline.ipynb`) with a
U-Net (`notebooks/02_unet.ipynb`).

## Planned work

- Full-resolution / crop-based training
- A second architecture
- A hyperparameter comparison
- A disc mask for the darkness-threshold baseline
