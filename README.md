# DUTransUNet
Dual-Decoder Transformer U-Net for Differential Interferogram Deformation Regions Segmentation

## Overview
This repository provides the implementation of DUTransUNet, a dual-decoder Transformer U-Net for deformation region segmentation in DInSAR differential interferograms. The repository includes the main training and testing scripts, sample data for quick verification, and split files for reproduction.

## Dataset
The experiments in the manuscript were conducted on the Hephaestus dataset.
## Dataset description
Hephaestus is a large-scale DInSAR dataset containing Sentinel-1 differential interferograms from 44 active volcanic regions worldwide. The dataset contains interferograms acquired between 2014 and 2021 and cropped into 224 × 224 patches.

## Dataset access
To facilitate rapid verification of the code, a subset of the Hephaestus dataset has been included in this repository for quick training and testing. The corresponding split files `train.txt` and `test.txt` are also provided.
A suggested directory layout is:

```text
data/
└── Hephaestus/
    ├── images/
    └── masks/

lists/
└── lists_hephaestus/
    ├── train.txt
    └── test.txt
```

The `data/` and `lists/` folders should be placed at the same directory level.
This subset is intended only for quick experiments and code verification. For larger-scale training and a more complete evaluation, users may obtain the full Hephaestus dataset from its official source and organize it following the same directory structure.
Please modify the dataset paths in the code if your local file structure is different.

## Usage

### 1. Environment

Please prepare an environment with Python 3.7, and then install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 2. Quick train/test

Run the training script on the provided subset:

```bash
python train.py
```

Run the testing script on the provided subset:

```bash
python test.py
```

The provided subset data can be used directly for a quick run of the training and testing pipeline.

## License
This project is released under the MIT License.
