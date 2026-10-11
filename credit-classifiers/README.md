# Credit Classifiers

This study compares logistic regression, Naive Bayes, random forest, and SVM.
It reports accuracy, macro F1, and balanced accuracy. SVM kernel and `C` are
selected through cross-validation on the training data.

Open the notebook from the repository root with:

```bash
python -m jupyter lab credit-classifiers/credit_classifiers.ipynb
```

The file `data/credit.pkl` is a legacy artifact from the course. Because pickle
files may execute code when loaded, use only the file already tracked in this
repository and never open an untrusted replacement.
