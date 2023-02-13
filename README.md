<img src="images/ips.png" width="600" />

# Iterative Patch Selection

Official implementation of "Iterative Patch Selection for High-Resolution Image Recognition" in PyTorch.

Iterative Patch Selection (IPS) is a simple patch-based approach that decouples the consumed memory from the input size and thus enables the efficient processing of high-resolution images without running out of memory. IPS works in two steps:  First, the most salient patches of an image are identified in no-gradient mode. Then, only selected patches are combined by a transformer-based patch aggregation module to train the network. 

**Accepted at ICLR 2023**

arXiv: https://arxiv.org/abs/2210.13007    
openreview: https://openreview.net/forum?id=QCrw0u9LQ7

## General usage

IPS is applied to 3 datasets: Traffic signs, Megapixel MNIST and CAMELYON16.  
The dataset can be set in `main.py`, by setting variable `dataset` to either traffic, mnist or camelyon.  
All other settings can be set in dataset specific .yml files in directory `config`.

Then, simply run: `python main.py`

## Notebook

The repo covers different data loading options (eager, eager sequential, lazy), positional encoding, tracking of efficiency metrics, single and multi-task learning. However, there is also simple example prepared as a Jupyter notebook (`ips_example.ipynb`), which can be loaded from Google Colab.

## Dataset specific considerations

**Traffic signs**: No specific considerations. The dataset will be downloaded automatically when running the main script.

**Megapixel MNIST**: Before training, the dataset needs to be created by running `data/megapixel_mnist/make_mnist.py`.  
For example: `python make_mnist.py --width 1500 --height 1500 dsets/megapixel_mnist_1500`.

**CAMELYON16**: Before training, the following steps need to be done:
1. Download the CAMELYON16 dataset (e.g. from the Grand Challenge website)

This repository implements IPS and demonstrates it on Megapixel MNIST.

First, the Megapixel MNIST dataset needs to be created:
    1. Create the path: data/megapixel_mnist/dsets/megapixel_mnist_1500
    2. Run data/megapixel_mnist/make_mnist.py, e.g.:
    python make_mnist.py --width 1500 --height 1500 dsets/megapixel_mnist_1500

All configurations can be found in config/mnist_config.yml
If you want to track computational efficiency metrics, set track_efficiency: True
If you want to train a model, set track_efficiency: False

Run: python main.py

If you are primarily interested in using IPS and/or the cross-attention MIL pooling
function in your own work, check architecture/ips_net.py and architecture/transformer.py
