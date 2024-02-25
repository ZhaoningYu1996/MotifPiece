# MotifPiece

## Publication

This project is described in detail in our paper titled "MotifPiece: A Data-Driven Approach for Effective Motif Extraction and Molecular Representation Learning". You can access the paper [here](https://arxiv.org/abs/2312.15387).

## Citing

If you find our work useful in your research, please consider citing:

```
@article{yu2023motifpiece,
  title={MotifPiece: A Data-Driven Approach for Effective Motif Extraction and Molecular Representation Learning},
  author={Yu, Zhaoning and Gao, Hongyang},
  journal={arXiv preprint arXiv:2312.15387},
  year={2023}
}
```


## MotifPiece Algorithm

![MotifPiece Algorithm](figures/motifpiece.png)

## Heterogeneous Graph Learning Module

![Heterogeneous Graph Learning Module](figures/HLM.png)

## Cross Datasets Learning Module

![Cross Datasets Learning Module](figures/CDL.png)

## Installation

### Environment setup

We highly recommend installing [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) for a simple environment setup and management.

Download our project:
```bash
git clone https://github.com/ZhaoningYu1996/MotifPiece.git
cd MotifPiece
```

Create a virtual environment with requirement packages:
```bash
conda env create -f environment.yml
```

Activate the virtual environment:
```bash
conda activate motifpiece
```

### Usage

To reproduce the results of running single dataset from the paper:
```bash
python main.py
```

To reproduce the results of cross datasets learning from the paper:
```bash
python cross_dataset_mol.py   # Datasets in MoleculeNet
python cross_dataset_ptc.py   # PTC datasets
```

To apply MotifPiece and extract motifs on a personal SMILES representation dataset, you can use MotifPiece class in motifpiece.py:
```bash
motifpiece = MotifPiece(*args)
```
In `*args`, you can setup `threshold`, `score_method`, and `merge_method`. You can also setup `train_indices` to only extract motifs from the training set.