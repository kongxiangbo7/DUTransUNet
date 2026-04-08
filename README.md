# DUTransUNet
Dual-Decoder Transformer U-Net for Differential Interferogram Deformation Regions Segmentation

# Dataset
The experiments in the manuscript were conducted on the Hephaestus dataset.
# Dataset description
Hephaestus is a large-scale DInSAR dataset containing Sentinel-1 differential interferograms from 44 active volcanic regions worldwide. The dataset contains interferograms acquired between 2014 and 2021 and cropped into 224 × 224 patches.
# Dataset access
Please obtain the Hephaestus dataset from its official source and organize it under the data/Hephaestus/ directory before training or testing.
A suggested directory layout is:
data/
└── Hephaestus/
    ├── images/
    ├── masks/
lists/
└──lists_hephaestus/
    ├── train.txt
    ├── test.txt



# Usage
## 1. Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 2.Train/Test
(1) Run the train script on Hephaestus dataset.
(2) Run the test script on Hephaestus dataset.
