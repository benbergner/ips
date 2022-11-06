This repository implements IPS and demonstrates it on Megapixel MNIST.

First, the Megapixel MNIST dataset needs to be created:
    1. Create the path: data/megapixel_mnist/dsets/megapixel_mnist_1500
    2. Run make_mnist.py in data/megapixel_mnist as follows:
    python make_mnist.py --width 1500 --height 1500 dsets/megapixel_mnist_1500

All configurations can be found in config/mnist_config.yml
If you want to track computational efficiency metrics,
set track_efficiency: True and track_epoch: 1
If you want to train a model, set track_efficiency: False

Run: python main.py