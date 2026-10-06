# Classificadores tabulares

O notebook canônico é `classificadores_credito.ipynb`. Ele compara regressão
logística, Naive Bayes, random forest e SVM, reportando acurácia, macro-F1 e
balanced accuracy. A seleção de kernel e `C` do SVM ocorre por validação
cruzada apenas no treino; o teste é consultado ao final.

```bash
python tools/execute_smoke_notebook.py 3_NaiveBayes/classificadores_credito.ipynb
```

Os notebooks anteriores das Partes 1–3 permanecem no histórico Git, mas foram
substituídos porque dependiam de pickles removidos e continham conclusões
estáticas. Os scripts em `Parte1`, `Parte2` e `Parte3` preservam os exercícios
didáticos atualizados.

`Parte2/credit.pkl` é o único artefato legado mantido. Ele não tem fonte bruta
equivalente no histórico. O notebook usa `load_trusted_legacy_pickle` e exige
`trusted=True`; nunca troque esse arquivo por um download não verificado.
