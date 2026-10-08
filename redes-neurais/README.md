# Redes neurais

Esta pasta preserva os projetos da atividade em notebooks separados:

- `rede_neural_tensorflow.ipynb` — rede neural densa com TensorFlow/Keras;
- `rede_neural_pytorch.ipynb` — variante da rede densa com PyTorch/skorch;
- `rede_convolucional_fashion_mnist.ipynb` — CNN para o Fashion-MNIST.

Os CSVs utilizados pelas redes densas estão em `dados/`. As dependências
pesadas são opcionais e separadas nos extras `tensorflow` e `pytorch` do
`pyproject.toml`. A CNN baixa o Fashion-MNIST na primeira execução.
