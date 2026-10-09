# Classificação com kNN

O notebook apresenta uma versão curta do experimento. O arquivo `exemplo.py`
executa os quatro conjuntos de dados originais, disponíveis em `dados/`.

A normalização faz parte da mesma `Pipeline` do classificador. Dessa forma, o
scaler é ajustado somente com os dados de treino de cada divisão. O valor de
`k`, a forma de votação e a medida de distância são escolhidos por validação
cruzada. O teste é usado apenas depois dessa escolha.

```bash
python classificacao-knn/exemplo.py
```

No conjunto D4, a primeira coluna é um identificador e não entra no treinamento.
