# Neural Networks

This directory contains three notebooks:

- `tensorflow_neural_network.ipynb`: binary classification with TensorFlow and Keras
- `pytorch_neural_network.ipynb`: the same study implemented with PyTorch and skorch
- `fashion_mnist_convolutional_network.ipynb`: dense and convolutional networks for Fashion-MNIST

The binary classification datasets are stored in `data/`. Fashion-MNIST is
downloaded by TensorFlow during the first execution.

These dependencies are larger and are not part of the base installation.
Install only the group required by the notebook you want to run:

```bash
# TensorFlow and the CNN notebook
python -m pip install "tensorflow>=2.16,<2.19" "scikeras>=0.13,<1"

# PyTorch
python -m pip install "torch>=2.3,<3" "skorch>=1,<2" "torchinfo>=1.8,<2"
```
