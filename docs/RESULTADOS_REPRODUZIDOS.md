# Resultados reproduzidos

Execução local em 2026-10-06, Python 3.12.14, scikit-learn 1.8.0 e sementes
fixadas conforme o código. Estes números foram gerados nesta reorganização;
podem variar com outras versões de dependências.

## kNN

O `MinMaxScaler` foi ajustado dentro de cada fold por uma `Pipeline`. `k`, pesos
e norma foram selecionados por validação cruzada no treino; o teste foi usado
somente após a seleção.

| Dataset | Melhor configuração | macro-F1 CV (média ± desvio) | Acurácia teste | Macro-F1 teste | Balanced accuracy teste |
| --- | --- | ---: | ---: | ---: | ---: |
| D1 | k=5, p=2, uniform | 0,9802 ± 0,0243 | 0,9600 | 0,9608 | 0,9608 |
| D2 | k=1, p=1, uniform | 0,9685 ± 0,0387 | 0,9833 | 0,9848 | 0,9861 |
| D3 | k=10, p=2, uniform | 0,6572 ± 0,0587 | 0,9400 | 0,9411 | 0,9412 |
| D4 | k=10, p=2, distance | 0,7075 ± 0,0361 | 0,6167 | 0,5532 | 0,5750 |

O D4 exclui a primeira coluna, tratada como identificador.

## Credit Data legado

Os modelos foram ajustados no treino preservado no `credit.pkl`; a seleção do
SVM foi feita por validação cruzada somente no treino. O teste preservado foi
usado para a tabela final.

| Modelo | Acurácia | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Regressão logística | 0,9460 | 0,8782 | 0,8757 |
| Naive Bayes | 0,9380 | 0,8454 | 0,8111 |
| Random forest | 0,9860 | 0,9680 | 0,9586 |
| SVM (C=10, RBF) | 0,9880 | 0,9735 | 0,9798 |

O artefato é legado e só deve ser carregado após verificação de procedência.
Esses resultados não substituem uma revisão da origem e da representatividade
do dataset.

## Smoke test do notebook kNN

O dataset sintético do notebook canônico produziu acurácia 0,6000, macro-F1
0,5986 e balanced accuracy 0,6010. Trata-se apenas de um teste de execução, não
de um benchmark.
