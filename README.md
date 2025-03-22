# Improving Group Fairness in Tensor Completion via Imbalance Mitigating Entity Augmentation

This repository contains the Python based implementation for the paper *Improving Group Fairness in Tensor Completion via Imbalance Mitigating Entity Augmentation, Dawon Ahn^, JunGi-Jang^, Evangelos E. Papalexkais (PAKDD 2025)*.

## Installation
To set up the environment, install the required packages:
```
- DotMap
- tensorly
- pytorch
```

## Configuration
There are configuration files for models in `config` directory. 
Modify `staff_{tf}.yaml` to adjust model hyperparameters. Configuration options include:
* `wnb_project`: project name for wandb (optional)
* `tf`: tensor factorization model (cpd or costco)
* `aug_tf`: selection of tf for augmentation (cpd or costco)
* `sampling`: augmentation type (knn)
* `aug_training`: generate augmentation or use saved augmentation
* `only_aug_save`: wheter to finish the script after saving augmentation

## Running Experiments
To train and evaluate the model, run `run.sh` script or run jupyter notebook in `demo` directory.
* Pre-trained tensor factorization models are saved in `output/{data}/{tf}` directories.
* Pre genereated augmentations are saved in `output/{data}/sampling`

## Dataset
This directory contains 
- tensor: `{name}.tensor` stored as COO format (i, j, k; v) 
- metadata: `{name}.json` including sensitive information

| **Name**      | **Mode**                       | **Nonzeros** | **Group** | **Majority** | **Minority** |
|------------------|--------------------------------|--------------|-----------|--------------|--------------|
|     **LastFM**   | **User** & Artist & Time | Interaction |**Gender** | Male         | Female             |
|                  | 853 & 2,964 & 1,586 | 143,107 |                            |93,316 | 49,791 
| **OULAD**        | **Student** & Module & Test | Score |   **Disability** | No           | Yes            |
|                  | 3,248 & 22 & 3 | 11,742       |              |           10,650       | 1,092        |
| **Chicago**      | Hour & **Area** & Crime | Crime Count | **Location** | South        | North            |
|                  | 24 & 77 & 32 | 42,097         |              |           23,723       | 18,374         |

## Directories
The repository follows this structure:
```
.
├── data/               # Dataset
├── src/                # Model immplementation
├── config/             # Configuration files
├── output/             # Output results and logs
├── demo/               # Jupyter notebook running demo.
└── README.md           # This documentation
```
