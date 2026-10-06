# Estudos de kNN

Os quatro datasets `.mat` são avaliados com o protocolo implementado em
`ml_studies.knn`:

- `MinMaxScaler` e `KNeighborsClassifier` permanecem na mesma `Pipeline`;
- o scaler é ajustado somente nos folds de treino;
- `k`, pesos e distância são escolhidos por validação cruzada no treino;
- o teste é usado uma única vez ao final;
- acurácia, macro-F1 e balanced accuracy são calculadas a partir da execução.

No dataset D4, a primeira coluna é removida por ser um identificador, não uma
característica preditiva.

```bash
python 2_kNN/demoD1.py
python 2_kNN/demoD2.py
python 2_kNN/demoD3.py
python 2_kNN/demoD4.py
```

Os resultados anteriormente escritos nos scripts foram removidos porque eram
conclusões estáticas e podiam divergir de uma nova execução.
