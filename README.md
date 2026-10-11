# Machine Learning Study Projects

This repository contains exercises and projects developed during the Machine
Learning course at FURB. It keeps the notebooks, supporting Python examples,
and datasets used throughout the course in one place.

The projects were completed as group assignments. The Git history preserves
the authorship of each contribution.

## Projects

| Directory | Topics |
| --- | --- |
| [`regression-studies`](regression-studies/) | Simple linear, multiple linear, and polynomial regression |
| [`knn-classification`](knn-classification/) | kNN classification across four datasets |
| [`credit-classifiers`](credit-classifiers/) | Comparison of Naive Bayes, random forest, logistic regression, and SVM |
| [`neural-networks`](neural-networks/) | Dense networks with TensorFlow and PyTorch, plus a CNN for Fashion-MNIST |

The final face identification project was moved to
[a separate repository](https://github.com/PHPacheco/reconhecimento-facial-lfw).

## Running the notebooks

Use 64-bit Python 3.12. Create the virtual environment and install the base
dependencies from the repository root:

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt

# Linux and macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
```

Open the notebook you want to study. For example:

```bash
.venv\Scripts\python.exe -m jupyter lab regression-studies/simple_linear_regression.ipynb
.venv\Scripts\python.exe -m jupyter lab regression-studies/multiple_linear_regression.ipynb
.venv\Scripts\python.exe -m jupyter lab regression-studies/polynomial_regression.ipynb
.venv\Scripts\python.exe -m jupyter lab knn-classification/knn_classification.ipynb
.venv\Scripts\python.exe -m jupyter lab credit-classifiers/credit_classifiers.ipynb
.venv\Scripts\python.exe -m jupyter lab neural-networks/tensorflow_neural_network.ipynb
.venv\Scripts\python.exe -m jupyter lab neural-networks/pytorch_neural_network.ipynb
.venv\Scripts\python.exe -m jupyter lab neural-networks/fashion_mnist_convolutional_network.ipynb
```

TensorFlow and PyTorch are optional because their packages are considerably
larger. Installation commands are available in the
[`neural-networks`](neural-networks/) README.

## Notes on the experiments

The kNN and SVM studies select hyperparameters using training data only. The
test set is reserved for the final evaluation. In addition to accuracy, the
classification studies report macro F1 and balanced accuracy when applicable.

Results depend on the data split, random seeds, and library versions. They
should be read as outcomes of course exercises, not as production performance
benchmarks.

The file `credit-classifiers/data/credit.pkl` is retained because an equivalent
tabular source is not available in the repository history. Pickle files may
execute code when loaded. Use only the version tracked in this repository and
never replace it with a file from an untrusted source.

## Authors

The repository history contains contributions from:

- [Pedro Krubniki Albuquerque (@pkalbuquerque)](https://github.com/pkalbuquerque)
- [Julia Welter (@juliawel)](https://github.com/juliawel)
- [Tiago Roberto de Melo (@TiagoRMelo)](https://github.com/TiagoRMelo)
- [Pedro Pacheco (@PHPacheco)](https://github.com/PHPacheco)
