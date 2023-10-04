# BERT-RNA


## Introduction

This is the samples for research "A multi-scale language-based deep learning model for interpretable prediction of RNA methylation across multiple species".

## Get Started

### basic dictionary
Initial parameters can be modified via `configuration/config.py`.

Datasets are included in `data/RNA_MS`.

### pretrain model
You should download pretrain model from relevant github repository.

For example, if you want to use [DNAbert](https://github.com/jerryji1993/DNABERT), you need to put them into the "pretrain" folder and rename the relevant choice in the model.

Here the zipped pre-trained models are also provided, you may refer to it as an example before getting your own model involved.

### Usage

``python main/train.py``
