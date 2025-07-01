# Simple Zero-Shot Image Dehazing with pretraining (SZID)

SZID is an image dehazing algorithm that leverages zero-shot learning principles and layer disentanglement techniques, surpassing the performance of existing models. The method uniquely operates on a single hazy image for both learning and inference, eliminating the need for extensive training datasets.

Authors: J.L. Lisani, J. Navarro, U. Untzilla

Paper submitted at ICIP2025

In this repository, we improve the inference time of SZID thanks to a pretraining done with the images of the I-HAZE dataset.

## Overview

This repository provides a PyTorch implementation of SZID.

## Credits

The codes in folders netZID and utilsZID come from https://github.com/XLearning-SCU/2020-TIP-ZID/

The rest of the code is based on https://github.com/jllisaniuib/SZID-Dehazing

## Code User Guide

### Dependencies

Create a virtual environment with all the needed dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.text
```

### Testing

If you want to dehaze an image using the pretrained weights, you can execute

```bash
python SZIDdehazing.py data/hazy.png output --path_weights=weights/default_weights.pth --num_iter=50
```

If you want to use SZID without pretraining, you can execute

```bash
python SZIDdehazing.py data/hazy.png output
```

In this case, the default number of iterations is 200.

The result will be saved in the folder 'results' as 'output_dehazed.png'. The input image will be cropped so its dimensions are multiple of 32 (requirement of the network input).

## Citation

If you find SZID useful in your research, please consider citing:

```
@article{SZID,
author = {U. Untzilla and J. Navarro and J.L. Lisani},
title = {{Simple Zero-Shot Image Dehazing}},
journal = {IEEE International Conference on Image Processing (ICIP)},
year = {2025},
}
```
