# Simple Zero-Shot Image Dehazing (SZID)

SZID is an image dehazing algorithm that leverages zero-shot learning principles and layer disentanglement techniques, surpassing the performance of existing models. The method uniquely operates on a single hazy image for both learning and inference, eliminating the need for extensive training datasets.

Authors: J.L. Lisani, J. Navarro, U. Untzilla

Paper submitted at ICIP2025

## Overview

This repository provides a PyTorch implementation of SZID.

## Credits

The codes in folders netZID and utilsZID come from https://github.com/XLearning-SCU/2020-TIP-ZID/

## Code User Guide

### Dependencies

Create a virtual environment with all the needed dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.text
```

### Training

To launch the training, execute

```
python SZIDdehazing.py --num_iter=200 --use_gpu=True
```

### Testing

If you want to dehaze an image you can execute

The result will be saved as 'output_dehazed.png'. The input image will be cropped so its dimensions are multiple of 32 (requirement of the network input). The cropped input image will be saved as 'output_original.png'.

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
