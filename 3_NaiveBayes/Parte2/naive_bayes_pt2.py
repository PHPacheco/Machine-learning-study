import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

# 1 - Importação dos dados Pré-Processados
with open('3_NaiveBayes/Parte2/risco_credito.pkl', 'rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)

# 2 - Algoritmo de Árvore de Decisão
# a) importar da biblioteca sklearn o pacote DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# b) Calcule a árvore de decisão, utilizando como critério a entropia. Coloque como nome da variável: arvore_risco_credito
arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')
arvore_risco_credito.fit(X_risco_credito, y_risco_credito)

# c) Utilize o feature_importances_ para retornar a importância de cada atributo. Qual possui o maior ganho de informação?
print(arvore_risco_credito.feature_importances_)

# d) Gere uma visualização da sua árvore de decisão utilizando o pacote tree da biblioteca do sklearn. OBS: Adicione cores, nomes para os atributos e para as classes.
from sklearn import tree
previsores = ['história', 'dívida', 'garantias', 'renda']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(arvore_risco_credito, feature_names=previsores, class_names=arvore_risco_credito.classes_, filled=True)
plt.show()

# e) FAZER A PREVISÃO
# i. história boa, dívida alta, garantia nenhuma, renda > 35
# ii. história ruim, dívida alta, garantia adequada, renda < 15
previsoes = arvore_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(previsoes)
