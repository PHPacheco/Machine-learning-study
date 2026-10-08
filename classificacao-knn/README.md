# Classificação com kNN

`classificacao_knn.ipynb` apresenta o protocolo canônico: o
`MinMaxScaler` permanece dentro da `Pipeline`, os hiperparâmetros são escolhidos
por validação cruzada no treino e o teste é consultado uma única vez.

Os quatro datasets originais estão em `dados/`. O exemplo Python executa todos
eles com o mesmo protocolo:

```bash
python classificacao-knn/exemplo.py
```

No dataset D4, a primeira coluna é removida porque representa um identificador,
não uma característica preditiva.
