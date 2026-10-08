# Classificadores de crédito

`classificadores_credito.ipynb` reúne os projetos de Naive Bayes, árvore e
random forest, regressão logística e SVM. As métricas são geradas pela execução
e incluem acurácia, macro-F1 e balanced accuracy.

O SVM escolhe kernel e `C` por validação cruzada somente no treino. O teste
preservado é consultado apenas na avaliação final.

```bash
python tools/execute_smoke_notebook.py classificadores-de-credito/classificadores_credito.ipynb
python classificadores-de-credito/exemplo.py
```

`dados/credit.pkl` é o único artefato legado mantido porque a fonte bruta
equivalente não existe no histórico. O loader exige `trusted=True`: nunca
substitua esse arquivo por um download não verificado.
