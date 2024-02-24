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