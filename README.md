# Overcoming the Diffraction Limit: Self-Supervised Terahertz Image Restoration using Self-Aligned Hyperspectral Images

Self-supervised deep learning framework for restoring terahertz (THz) images beyond the
diffraction limit, using **self-aligned hyperspectral (SAHS)** images acquired from a
THz time-domain spectroscopy (THz-TDS) system — **without any external training dataset
or artificial image degradation**.

The key idea is to learn a mapping between THz images at different frequencies. Because
the diffraction limit and noise level of a THz-TDS system vary with frequency, this
frequency-to-frequency mapping lets low-frequency (blurry) images mimic the sharpness of
high-frequency images, and high-frequency (noisy) images mimic the cleanliness of
low-frequency images.

## Method

The framework has three stages:

| Stage | Name | Description |
|-------|------|-------------|
| 1 | **O2O** (One-to-One) | Train a network to map an image at frequency α to an image at frequency β. Setting α&lt;β sharpens (deblurs) low-frequency images; α&gt;β denoises high-frequency images. |
| 2 | **RE** (Recurrent) | Repeat O2O, using the previous target as the new source and the previous output as the new target, to progressively refine the result. |
| 3 | **ZSSR** (Zero-Shot Super-Resolution) | Increase the image resolution from a single image, by learning internal recurrence (based on [Shocher et al., 2018](https://arxiv.org/abs/1712.06087)). |

Both **intra-data** inference (on the images used for training) and **cross-data**
inference (on new, unrelated THz images) are supported.

## Repository structure

```
net.py                       # ZSSRNet (8-conv CNN, global residual skip)
data.py                      # DataSampler (paired frequency images; scale-aware for ZSSR)
source_target_transforms.py  # Paired augmentations (rotate / flip / crop)
train.py / train.sh          # Training entry point and example runs
test.py  / test.sh           # Inference entry point and example runs
IQA/                         # Image-quality metrics (sharpness, noise level, MTF, ...)
contrast/                    # Contrast metric
axial_graph/                 # Beam-width (FWHM) / axial line-profile analysis
hyperspectral/               # THz SAHS images (per-frequency PNGs)
paper/                       # Manuscript (LaTeX)
```

## Requirements

- Python 3, PyTorch, torchvision
- numpy, Pillow, tqdm, wandb (training logs; run `wandb offline` or set `WANDB_MODE=disabled` to skip)

## Data

THz SAHS images are stored as per-frequency PNGs, e.g.

```
hyperspectral/centernorm_step0.1_reshapev2.0x5_shift1/resol1951_2_e_2.0THz.png
```

Each image is a single THz frequency slice (0.1–3.0 THz) of the USAF 1951 resolution test
chart, obtained by FFT of the time-domain THz-TDS signal at every pixel. Pixel intensities
are normalized to 0–255.

## Usage

The network is trained with an L1 pixel loss and a global residual skip connection
(`output = model(input) + input`), Adam (lr 1e-5, decayed ×0.1 after 10,000 iters),
15,000 iterations, and 96×96 random crops.

### 1. O2O — deblurring / denoising

Train a mapping from frequency α to β and run inference:

```bash
# Deblur: map 1.5 THz (source) -> 2.0 THz (target)
python train.py \
    --img    hyperspectral/centernorm_step0.1_reshapev2.0x5_shift1/resol1951_2_e_1.5THz.png \
    --target hyperspectral/centernorm_step0.1_reshapev2.0x5_shift1/resol1951_2_e_2.0THz.png \
    --model ZSSR --exp o2o

python test.py \
    --ckpt_path checkpoints_main/ZSSR_o2o/resol1951_2_e_1.5_to_2.0THz_latest.pt \
    --save_dir o2o --model ZSSR
```

For **denoising**, set the source frequency higher than the target (α&gt;β).
For **cross-data** inference, pass `--test_img <path>` to run the trained model on a new image.

### 2. RE — recurrent refinement

Re-run O2O using the previous target as the source and the previous inference result as
the target (see the recurrent examples in `train.sh` / `test.sh`).

### 3. ZSSR — super-resolution

Apply ZSSR to an image (typically an O2O result) with `--scale` to increase its
resolution. The input is bicubically upscaled by `scale` and the network restores the
high-frequency detail at the enlarged resolution:

```bash
# 2x super-resolution (282x282 -> 564x564)
python train.py --img <image>.png --target <image>.png --model ZSSR --exp zssr --scale 2
python test.py  --test_img <image>.png \
    --ckpt_path checkpoints_main/ZSSR_zssr/<...>_latest.pt \
    --save_dir zssr --model ZSSR --scale 2
```

`--scale 1` (default) keeps the original resolution, reproducing the plain O2O / RE behavior.

## Image quality assessment

Metrics reported in the paper live under `IQA/` and `contrast/`:
sharpness (`IQA/acutance.py`), noise level (`IQA/noise_level_estimation.py`),
contrast (`contrast/calc_contrast.py`), MTF / spatial-frequency analysis
(`IQA/MFA.ipynb`), and beam-width FWHM (`axial_graph/`).

## Acknowledgements

The network and training loop build upon
[Jacob Gildenblat's pytorch-zssr](https://github.com/jacobgil/pytorch-zssr), with
modifications and enhancements for THz self-aligned hyperspectral image restoration.
