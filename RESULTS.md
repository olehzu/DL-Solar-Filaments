# Results: full-resolution training (part 1) and a second architecture (part 2)

This file summarizes what notebook 03 (`notebooks/03_unet_crops.ipynb`) found for parts 1 and 2 of the plan in our
midterm report. Part 3 (hyperparameter comparison) is still to come.

## What we tested

At the midterm our best model was a U-Net (ResNet34 encoder, pretrained on ImageNet) trained on images shrunk from
2048 x 2048 to 512 x 512. Shrinking makes thin filaments one or two pixels wide or removes them, so the plan was:

1. **Part 1, resolution.** Train the same U-Net on 512 x 512 crops cut from the full 2048 x 2048 images and predict a
   full image tile by tile, then compare it with the U-Net trained on shrunk images.
2. **Part 2, architecture.** Train a different segmentation model with the same setup and compare it with the U-Net.

| Run | Model | Training input |
|---|---|---|
| `resized_512` | U-Net | whole image shrunk to 512 x 512 (the midterm setting) |
| `crops_2048` | U-Net | 512 x 512 crops of the full 2048 image, half of them placed on a filament |
| `unetpp_crops_2048` | UNet++ (same ResNet34 encoder, different decoder) | same as `crops_2048` |

Everything else is shared: the day-based split from the midterm (fingerprint `b66a782d2360`), weighted BCE + soft Dice
loss, Adam at 3e-4, batch 8, 15 epochs, the same augmentation, the best epoch chosen by validation Dice, the threshold
tuned on validation, and one scoring of the test set (173 images, 1164 annotated filaments). All runs trained on a
Tesla T4 in Google Colab.

## Results on the test set

Pooled over all pixels of the 173 test images. "@2048" uses masks at the original size, "@512" uses masks at 512 x 512
so it can be compared with the midterm.

| Run | IoU @2048 | Dice @2048 | Precision @2048 | Recall @2048 | IoU @512 | Dice @512 | Train time | Test time per image |
|---|---|---|---|---|---|---|---|---|
| Midterm U-Net (notebook 02) | - | - | - | - | 0.5331 | 0.6955 | 7.7 min | - |
| `resized_512` | 0.5212 | 0.6853 | **0.6792** | 0.6914 | 0.5364 | 0.6983 | 26.5 min | 0.03 s |
| `crops_2048` | 0.5130 | 0.6781 | 0.6493 | 0.7096 | 0.5199 | 0.6841 | 35.7 min | 0.24 s |
| `unetpp_crops_2048` | **0.5269** | **0.6902** | 0.6585 | **0.7251** | **0.5370** | **0.6988** | 77.3 min | 0.55 s |

Parameters: U-Net 24.4M, UNet++ 26.1M. Thresholds tuned on validation (@2048): 0.80 for `resized_512`, 0.50 for both
crop runs.

### Recall by filament width

For every annotated test filament we measure the share of its pixels that the prediction covers, and group the
filaments into four equal groups by estimated width (2 x area / perimeter, in pixels at 2048).

| Run | < 7.4 px (thinnest) | 7.4 - 9.2 px | 9.2 - 12.0 px | > 12.0 px (thickest) | Filaments missed completely |
|---|---|---|---|---|---|
| `resized_512` | 0.540 | 0.608 | 0.630 | 0.730 | 17.9 % |
| `crops_2048` | **0.669** | **0.677** | 0.702 | 0.749 | **7.0 %** |
| `unetpp_crops_2048` | 0.637 | 0.668 | **0.707** | **0.767** | 10.4 % |

Each group holds 291 filaments.

## What we found

**Part 1: full resolution finds many more thin filaments, but the overall score stays the same.**
Training on full-resolution crops raised the recall of the thinnest filaments from 0.54 to 0.67 and cut the share of
filaments that are missed completely from 17.9 % to 7.0 %. That is the problem the crops were meant to fix. The pooled
Dice did not improve (0.678 against 0.685), because the crop model also marks more pixels that are not filament
(precision 0.65 against 0.68). Two likely reasons: half of the training crops are placed on a filament, so the model
sees filaments more often than they really occur, and a 512 x 512 crop shows less of the Sun around a dark feature.

**The pipeline reproduces the midterm.** `resized_512` gets Dice 0.698 at 512, next to 0.6955 in the midterm report, so
the new notebook agrees with notebook 02.

**Part 2: UNet++ is slightly better overall, but not on thin filaments.**
UNet++ has the best pooled scores (Dice @2048 0.690, about 0.012 above the crop U-Net), with both higher precision and
higher recall. We expected its extra skip connections to help most on thin filaments. They did not: on the thinnest
group it is slightly below the crop U-Net (0.64 against 0.67), and its gain comes from the thicker filaments and from
fewer false positives. It also takes about twice as long to train and three times as long per test image.

## Caveats

- Every number comes from a single training run with one seed. Differences of about 0.01 Dice, such as UNet++ against
  the crop U-Net, may be noise. The thin-filament gain from the crops (+0.13 recall) is large enough to take seriously.
- The crop run differs from `resized_512` in two ways at once: the resolution and the rule that puts half of the crops
  on a filament.
- UNet++ uses the learning rate, loss and number of epochs chosen for the U-Net.
- The scores come from our own test split, not from the Kaggle leaderboard.

## Still to run

- DeepLabV3+ (`deeplabv3p_crops_2048`) and Flat U-Net (`flatunet_crops_2048`) are implemented in notebook 03 but not
  trained yet (Colab GPU limits). They are optional for part 2.
- Part 3 can start from `crops_2048`. A useful first run is `crops_2048` with `pos_crop_prob=0.0`, which shows whether
  the filament-focused crops cause the extra false positives.

## Where the numbers come from

The weights and result files are in `MyDrive/DL-Solar-Filaments/outputs` (git-ignored, not in this repository):
`results_<run>.json` holds the configuration, training history, thresholds, test metrics and per-filament recall of
each run. The tables above are read from those three files. The recall-by-width groups use the same quartile edges as
the notebook's `recall_by_width.csv`.
