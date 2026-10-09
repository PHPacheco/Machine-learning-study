# Classificadores de crédito

Este estudo compara regressão logística, Naive Bayes, random forest e SVM. As
métricas calculadas são acurácia, macro-F1 e balanced accuracy. No caso do SVM,
o kernel e o valor de `C` são escolhidos por validação cruzada nos dados de
treino.

O notebook contém a explicação do exercício. O mesmo experimento pode ser
executado pelo terminal:

```bash
python classificadores-de-credito/exemplo.py
```

O arquivo `dados/credit.pkl` é um artefato antigo da disciplina. Como `pickle`
pode executar código durante a leitura, use somente o arquivo que já faz parte
deste histórico e não abra cópias de origem desconhecida.
