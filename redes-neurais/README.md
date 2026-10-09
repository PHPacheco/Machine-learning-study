# Redes neurais

A pasta contém três notebooks:

- `rede_neural_tensorflow.ipynb`: classificação binária com TensorFlow e Keras;
- `rede_neural_pytorch.ipynb`: o mesmo estudo usando PyTorch e skorch;
- `rede_convolucional_fashion_mnist.ipynb`: redes densa e convolucional para o
  Fashion-MNIST.

Os dados do estudo de classificação binária estão em `dados/`. O Fashion-MNIST
é baixado pela própria biblioteca na primeira execução.

As dependências dessas atividades são maiores e não fazem parte da instalação
básica. Instale somente o conjunto do notebook que pretende executar:

```bash
# TensorFlow e CNN
python -m pip install "tensorflow>=2.16,<2.19" "scikeras>=0.13,<1"

# PyTorch
python -m pip install "torch>=2.3,<3" "skorch>=1,<2" "torchinfo>=1.8,<2"
```
