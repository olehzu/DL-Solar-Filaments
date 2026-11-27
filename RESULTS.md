# Results: full-resolution training (part 1), a second architecture (part 2) and hyperparameters (part 3)

This file summarizes what notebook 03 (`notebooks/03_unet_crops.ipynb`) found for the three parts of the plan in our
midterm report. Parts 1 and 2 are described first; part 3 (UNet++ with different hyperparameters) has its own section
further down.

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
| `deeplabv3p_crops_2048` | DeepLabV3+ (same ResNet34 encoder, different decoder) | same as `crops_2048` |
| `flatunet_crops_2048` | Flat U-Net (Zhu et al., 2025), no encoder, trained from scratch | same as `crops_2048` |

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
| `deeplabv3p_crops_2048` | 0.5064 | 0.6723 | 0.6345 | 0.7149 | 0.5166 | 0.6813 | 27.0 min | 0.18 s |
| `flatunet_crops_2048` | 0.3547 | 0.5237 | 0.5203 | 0.5271 | 0.3571 | 0.5263 | 44.3 min | 0.33 s |

Parameters: U-Net 24.4M, UNet++ 26.1M, DeepLabV3+ 22.4M, Flat U-Net 0.26M. Thresholds tuned on validation (@2048):
0.80 for `resized_512`, 0.50 for `crops_2048` and `unetpp_crops_2048`, 0.60 for `deeplabv3p_crops_2048` and 0.45 for
`flatunet_crops_2048`. Best epoch by validation Dice: 7 of 15 for DeepLabV3+, 14 of 15 for Flat U-Net.

### Recall by filament width

For every annotated test filament we measure the share of its pixels that the prediction covers, and group the
filaments into four equal groups by estimated width (2 x area / perimeter, in pixels at 2048).

| Run | < 7.4 px (thinnest) | 7.4 - 9.2 px | 9.2 - 12.0 px | > 12.0 px (thickest) | Filaments missed completely |
|---|---|---|---|---|---|
| `resized_512` | 0.540 | 0.608 | 0.630 | 0.730 | 17.9 % |
| `crops_2048` | **0.669** | **0.677** | 0.702 | 0.749 | 7.0 % |
| `unetpp_crops_2048` | 0.637 | 0.668 | 0.707 | **0.767** | 10.4 % |
| `deeplabv3p_crops_2048` | 0.638 | 0.676 | **0.733** | 0.766 | **5.0 %** |
| `flatunet_crops_2048` | 0.446 | 0.506 | 0.589 | 0.576 | 6.6 % |

Each group holds 291 filaments. A filament counts as missed completely when the prediction covers none of its pixels.

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

**DeepLabV3+ finds the most filaments but draws them less precisely.**
DeepLabV3+ misses only 5.0 % of the test filaments completely, the fewest of all runs, and has the best recall in the
9.2 - 12.0 px group (0.73). Its pooled scores are the lowest of the ResNet34 models (Dice @2048 0.672, 0.006 below the
crop U-Net and 0.018 below UNet++), and the loss comes from precision (0.63, the lowest of the three). This fits its
design: the wide context of the ASPP module helps it notice a filament, but the prediction is made at 1/4 resolution
and upsampled, so the outline is coarser and spills over the edges. As expected, the thinnest group is where it gives
up the most against the crop U-Net (0.64 against 0.67). It is also the cheapest of the crop models to run: 27 min of training and
0.18 s per test image, against 35.7 min and 0.24 s for the crop U-Net. Its validation Dice peaked at epoch 7 and stayed
between 0.64 and 0.68 afterwards, so more epochs would probably not change much.

**Flat U-Net is far behind after 15 epochs, and it is not yet finished training.**
Flat U-Net reaches Dice @2048 0.524, about 0.15 below the other crop models, with lower precision and recall alike and
much lower recall in every width group. It still finds at least part of most filaments (6.6 % missed completely), but
covers a smaller share of each. With 0.26M parameters it is about 1 % the size of the U-Net, yet it is not faster here:
it keeps 32 channels at full resolution and has an attention block at every level, so training took 44.3 min and
prediction 0.33 s per image, both slower than the U-Net. The comparison is not fair to it yet. It is the only model
without ImageNet pretraining, its best epoch was 14 of 15 and its training loss was still falling, and the authors
trained it for 100 epochs with a learning-rate schedule. The paper's validation Dice of 0.82 was also measured on their
own data, not on MAGFiLO.

**Part 2 summary.** No architecture clearly beats the crop U-Net. UNet++ is slightly better overall, DeepLabV3+ is
slightly worse overall but misses the fewest filaments and is the fastest crop model, and the crop U-Net is still best on the
thinnest filaments. The three ResNet34 models are all within about 0.02 Dice of each other, so with this encoder and
training setup the choice of decoder changes little, and single-seed differences this small may be noise.

## Part 3: UNet++ with different hyperparameters

UNet++ had the best pooled scores in part 2, so part 3 trains it again with one setting changed at a time and compares
each run with `unetpp_crops_2048` from part 2. Everything not listed is unchanged (crops of 512 x 512 from the 2048
image, Adam, batch 8, 15 epochs, the same split, loss, augmentation, threshold tuning and test set).

| Run | What changes against `unetpp_crops_2048` |
|---|---|
| `unetpp_pos025_crops_2048` | a quarter of the training crops placed on a filament instead of half (`pos_crop_prob` 0.5 to 0.25) |
| `unetpp_pos0_crops_2048` | no crops placed on a filament, all positions random (`pos_crop_prob` 0) |
| `unetpp_seed1_crops_2048` | only the training seed (1 instead of 42): weights of the new layers, crop positions, augmentation and batch order. The split is the same |
| `unetpp_lr1e-4_crops_2048` | learning rate 1e-4 instead of 3e-4 |
| `unetpp_lr1e-3_crops_2048` | learning rate 1e-3 instead of 3e-4 |

The seed run changes no hyperparameter. It measures how much the scores move from training randomness alone, which is
the yardstick for the other four.

### Results on the test set

| Run | IoU @2048 | Dice @2048 | Precision @2048 | Recall @2048 | IoU @512 | Dice @512 | Val Dice @2048 | Best epoch | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| `unetpp_crops_2048` (part 2) | 0.5269 | 0.6902 | 0.6585 | **0.7251** | **0.5370** | **0.6988** | - | - | 0.50 |
| `unetpp_pos025_crops_2048` | **0.5276** | **0.6908** | **0.6667** | 0.7166 | 0.5360 | 0.6979 | **0.6956** | 12 | 0.50 |
| `unetpp_pos0_crops_2048` | 0.5207 | 0.6848 | 0.6610 | 0.7105 | 0.5357 | 0.6977 | 0.6909 | 15 | 0.60 |
| `unetpp_seed1_crops_2048` | 0.5159 | 0.6807 | 0.6476 | 0.7172 | 0.5242 | 0.6878 | 0.6910 | 15 | 0.65 |
| `unetpp_lr1e-4_crops_2048` | 0.5217 | 0.6857 | 0.6628 | 0.7103 | 0.5305 | 0.6933 | 0.6916 | 7 | 0.50 |
| `unetpp_lr1e-3_crops_2048` | 0.5114 | 0.6768 | 0.6430 | 0.7142 | 0.5211 | 0.6851 | 0.6815 | 7 | 0.45 |

### Recall by filament width

Same width groups as above (quartile edges 7.4, 9.2 and 12.0 px, 291 filaments each).

| Run | < 7.4 px (thinnest) | 7.4 - 9.2 px | 9.2 - 12.0 px | > 12.0 px (thickest) | Filaments missed completely |
|---|---|---|---|---|---|
| `unetpp_crops_2048` (part 2) | 0.637 | 0.668 | 0.707 | 0.767 | 10.4 % |
| `unetpp_pos025_crops_2048` | 0.668 | 0.672 | 0.721 | 0.741 | 8.2 % |
| `unetpp_pos0_crops_2048` | 0.654 | 0.658 | 0.707 | 0.746 | 8.9 % |
| `unetpp_seed1_crops_2048` | **0.686** | 0.678 | **0.731** | **0.769** | **4.9 %** |
| `unetpp_lr1e-4_crops_2048` | **0.686** | **0.692** | 0.726 | 0.741 | 5.2 % |
| `unetpp_lr1e-3_crops_2048` | 0.679 | 0.685 | 0.727 | 0.753 | 5.2 % |

### What we found

**Training randomness alone moves Dice by about 0.01.** The seed run has exactly the settings of `unetpp_crops_2048`
and scores Dice @2048 0.681 against 0.690. All five part 3 runs land between 0.677 and 0.691, a spread only about 1.5
times this seed difference. With one run per setting, none of the changes can be said to improve or hurt the model.

**The thin-filament numbers move even more with the seed.** The seed run misses 4.9 % of the test filaments completely
against 10.4 % for the part 2 run, and its recall on the thinnest group is 0.686 against 0.637. Differences of this
size in parts 1 and 2 (for example UNet++ against the crop U-Net on thin filaments, or DeepLabV3+ missing the fewest
filaments) should therefore be read as possible noise too. The part 1 gain from training on crops (thinnest-group
recall 0.54 to 0.67, missed filaments 17.9 % to 7.0 %) is still larger than this.

**Fewer filament-centred crops: no clear effect.** In part 1 we suspected that placing half of the crops on a filament
causes extra false positives. For UNet++ this did not show up: precision is 0.659 with half the crops on filaments,
0.667 with a quarter and 0.661 with none, all within noise. `pos_crop_prob` 0.25 gave the best pooled scores of part 3
(Dice 0.691, IoU 0.528, precision 0.667, best validation Dice 0.696), but only 0.0006 Dice above the part 2 run. With
no filament-centred crops at all, the first epochs were clearly worse (validation Dice 0.47 after epoch 1 against 0.64
with a quarter), but after 15 epochs the model had caught up.

**Learning rate: 3e-4 is a good choice; 1e-3 is too high.** Learning rate 1e-3 gave the lowest scores of part 3 (Dice
0.677, precision 0.643, best validation Dice 0.682) and the most unstable training (validation Dice between 0.59 and
0.68 from epoch to epoch). Learning rate 1e-4 is within noise of 3e-4 (Dice 0.686); its validation Dice peaked at
epoch 7 and then drifted down while the training loss kept falling, so a lower learning rate did not need more than 15
epochs. Both learning-rate runs and the seed run miss about 5 % of the filaments completely, so the 10.4 % of the part
2 run looks like the unlucky one rather than a property of 3e-4.

**Part 3 summary.** None of the five changes clearly beats the UNet++ settings from part 2. Within the ranges we
tried, the share of filament-centred crops and a learning rate between 1e-4 and 3e-4 make no measurable difference,
and 1e-3 is worse. The best part 3 run (`pos_crop_prob` 0.25, Dice @2048 0.691) is a reasonable final model, but its
lead over the part 2 settings is not significant. The largest effect we measured in part 3 is the run-to-run noise,
which means telling apart settings this close would need several training runs with different seeds per setting.

## Caveats

- Every number comes from a single training run per setting. The part 3 seed run shows that training randomness alone
  moves Dice @2048 by about 0.01 and the share of filaments missed completely by about 5 percentage points, so
  differences of that size between any two runs in this file may be noise. The thin-filament gain from the crops
  (+0.13 recall) is larger than that.
- The crop run differs from `resized_512` in two ways at once: the resolution and the rule that puts half of the crops
  on a filament.
- UNet++ and DeepLabV3+ use the learning rate, loss and number of epochs chosen for the U-Net. Part 3 checks the
  learning rate and crop sampling for UNet++ only, one setting at a time, so interactions between settings are not
  tested.
- `unetpp_lr1e-4_crops_2048` (from epoch 6) and `unetpp_lr1e-3_crops_2048` ran in a second Colab session, after the
  first one hit the GPU limit. Their scores are comparable with the other runs (same notebook, data, split, settings
  and type of GPU), but their timings are not: in that session every epoch took about 350 s instead of 308 s,
  including the epochs of `unetpp_lr1e-4_crops_2048` that resumed from its epoch-5 checkpoint, so the slowdown came
  from the session and not from the learning rate. Train time and test time per image are therefore left out of the
  part 3 comparison.
- Flat U-Net is the only model trained from scratch, and 15 epochs at learning rate 3e-4 is far less training than the
  paper used. Its numbers say how it does under our budget, not how good the architecture can be.
- The scores come from our own test split, not from the Kaggle leaderboard.

## Where the numbers come from

The weights and result files are in `MyDrive/DL-Solar-Filaments/outputs` (git-ignored, not in this repository):
`results_<run>.json` holds the configuration, training history, thresholds, test metrics and per-filament recall of
each run. The recall-by-width groups use the same quartile edges as the notebook's `recall_by_width.csv`.

For the five part 3 runs, `results_<run>.json` is in this repository under [`results/<run>/`](results/). They ran on a
Tesla T4 in Colab with the same software versions as below, using notebook 03 from the `part3-unetpp-grid` branch.

For `deeplabv3p_crops_2048` and `flatunet_crops_2048` the result files are also in this repository, under
[`results/<run>/`](results/): `results_<run>.json`, `comparison_test.csv`, `recall_by_width.csv` and `curves.png` (train
loss and validation Dice per epoch). Both ran on a Tesla T4 in Colab (Python 3.13, torch 2.14.1, smp 0.6.0), each in its
own session with `RUNS` set to the one run, using notebook 03 as it is on `main`. `results/split_seed42.json` is the
split they used (fingerprint `b66a782d2360`, the same as all other runs). The model weights (90 MB for DeepLabV3+, 1 MB
for Flat U-Net) and the `qualitative_test.png` figures stay on Drive: weights are git-ignored, and the figures show
MAGFiLO images, which we do not publish.
