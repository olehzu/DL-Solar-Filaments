# Solar Filament Segmentation (COSC 50523 Deep Learning, TCU)

Course project on segmenting solar filaments in full-disk solar images from the Kaggle competition (no longer competing)
"Solar Filament Segmentation Challenge 2026" (MAGFiLO dataset, not included in this repo). We compare a
darkness-threshold baseline and a per-pixel logistic regression (`notebooks/01_logreg_baseline.ipynb`) with a
U-Net (`notebooks/02_unet.ipynb`). `notebooks/03_unet_crops.ipynb` trains the same U-Net on 512 x 512 crops of the
full 2048 x 2048 images and compares it with a U-Net trained on whole images resized to 512. It also trains UNet++ and
DeepLabV3+ on the same crops as a second architecture.

## Notebook 03: crop-based training at full resolution

Open it in Colab with a GPU runtime. The first cells download the data with the same Kaggle cell as notebooks 01 and 02
and make the same day-based split (same code and seed). Every image and mask is then decoded once into `cache/`, which
takes about 3.5 GB of disk. Training reads one crop at a time from there, so the full-size images never have to sit in
memory. Four models are trained one after the other. `resized_512` and `crops_2048` are U-Nets that differ only in
resolution. `unetpp_crops_2048` (UNet++) and `deeplabv3p_crops_2048` (DeepLabV3+) use the `crops_2048` settings and the
same ResNet34 encoder with a different decoder. All runs share the split, loss, optimizer, augmentation and number of
training steps, and all are scored against masks at 2048 x 2048 and at 512 x 512. We have not timed a full run on Colab
yet. Our rough guess is an hour or more per crop run on a T4, so it may help to run one model per session with the
`RUNS` environment variable. If UNet++ runs out of GPU memory, add `batch=4` to its entry in `RUNS`.

The notebook writes the best weights of each run, `results_<run>.json` with the configuration and metrics,
`comparison_test.csv`, `recall_by_width.csv`, `curves.png` and `qualitative_test.png`. On Colab the first cell mounts
Google Drive and puts all of it in `MyDrive/DL-Solar-Filaments/outputs`, because Colab deletes `/content` when the
session ends and the files are git-ignored. Outside Colab they go to `outputs/`. A checkpoint is saved after every epoch.
If the session drops, run the notebook again from the top. Finished runs are skipped, and the interrupted one continues
after its last saved epoch with the same crops it would have drawn anyway. A checkpoint is only resumed when the run's
settings match, so give each new configuration its own name in `RUNS`. `RESUME=0` starts over, `SAVE_TO_DRIVE=0` keeps
everything local, and `REUSE_WEIGHTS=1` loads saved weights and only evaluates. The other settings are listed at the
top of the notebook.

To run it locally you need Python 3.9 or newer with numpy, pillow, matplotlib, pandas, torch (2.3 or newer) and
segmentation-models-pytorch. We checked it with Python 3.13, torch 2.15 and segmentation-models-pytorch 0.6.0. On macOS
and Windows the DataLoaders run in the main process (`NUM_WORKERS=0`), because Jupyter there cannot pass classes
defined in a notebook to worker processes.

## Planned work

- Full-resolution / crop-based training (notebook 03, waiting on a full Colab run)
- A second architecture: UNet++ and DeepLabV3+ (notebook 03, waiting on a Colab run)
- A hyperparameter comparison
- A disc mask for the darkness-threshold baseline
