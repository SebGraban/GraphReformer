# GraphReformer

This repository is the official implementation of GraphFormer, a generative model using Transformer based autoregressive models.

Most of the code has been adapted from [GraphGen](https://github.com/idea-iitd/graphgen)

## Installation

Pytorch and pip installation in conda. Change cuda version as per your GPU hardware support.

Install dependencies via: 

```bash
pip install -r requirements.txt
```

[Boost](https://www.boost.org/) (Version >= 1.70.0) and [OpenMP](https://www.openmp.org/) are required for compling C++ binaries. Run `build.sh` script in the project's root directory.

```bash
./build.sh
```

## Test run

```bash
python3 main.py
```

## Code Description

The code is designed to all be ran using main.py, this includes data prep, training, and analysis producing a final CSV with all of the metrics for the different datasets that have been included.


- Everything under graphgen comes from the graphgen repository for further details please see the link above
- dataset includes all of the data along with the data preparation, including padding and transforming the data into a tensor.
- model includes the models themselves in graphreformer.py and the train and prediction loops found in train.py

