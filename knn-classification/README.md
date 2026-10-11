# kNN Classification

The notebook presents a compact, reproducible version of the experiment. The
four original datasets remain available in `data/` for further study.

Normalization is part of the same `Pipeline` as the classifier, so the scaler
is fitted only on the training data for each split. Cross-validation selects
the value of `k`, the voting strategy, and the distance metric. The test set is
used only once after model selection.

```bash
python -m jupyter lab knn-classification/knn_classification.ipynb
```

In dataset D4, the first column is an identifier and is excluded from training.
