import matplotlib
matplotlib.use('TkAgg')

import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import utils

def main():
    CAMINHO_BASE = 'risco_credito.pkl'

    print("PARTE 3: Regressão Logística")

    df = utils.RiscoCredito.carregar(CAMINHO_BASE)

    print(f"\nBase com: {len(df)} registros\n")
    print(df.to_string(index=True))

    df = utils.RiscoCredito.remover_moderado(df)

    print(f"\nBase com: {len(df)} registros\n")
    print(df.to_string(index=True))

    X = utils.RiscoCredito.encode_previsores(df)
    y, label_encoder = utils.RiscoCredito.encode_classe(df)

    print(f"\nEncoder - Classes codificadas: {dict(enumerate(label_encoder.classes_))}")
    print(f"   (0 = alto  |  1 = baixo)")
    print(f"\nX (previsores):\n{X}")
    print(f"\ny (classe codificada):\n{y}")

    modelo = LogisticRegression(random_state=1)
    modelo.fit(X, y)

    print("\n\nParâmetros do modelo de treino")
    print(f"\nIntercepto (B0) — modelo.intercept_:\n   {modelo.intercept_}")
    print(f"\nCoeficientes — modelo.coef_:\n   {modelo.coef_}")

    print("\n\nPredições para novos clientes")

    # a) história boa(2), dívida alta(1), garantias nenhuma(0), renda >35(2)
    cliente_a = utils.RiscoCredito.novo_cliente(historia=2, divida=1, garantias=0, renda=2)
    pred_a = label_encoder.inverse_transform(modelo.predict(cliente_a))
    print(f"\nCliente A:")
    print(f"   história=boa | dívida=alta | garantias=nenhuma | renda=>35")
    print(f"   ➜  Risco previsto: '{pred_a[0]}'")   # baixo

    # b) história ruim(0), dívida alta(1), garantias adequada(1), renda <15(0)
    cliente_b = utils.RiscoCredito.novo_cliente(historia=0, divida=1, garantias=1, renda=0)
    pred_b = label_encoder.inverse_transform(modelo.predict(cliente_b))
    print(f"\nCliente B:")
    print(f"   história=ruim | dívida=alta | garantias=adequada | renda=<15")
    print(f"   ➜  Risco previsto: '{pred_b[0]}'")   # alto

    # distribuição das classes
    contagem = df['risco'].value_counts().reset_index()
    contagem.columns = ['risco', 'quantidade']

    cores = contagem['risco'].map({'alto': '#E74C3C', 'baixo': '#2ECC71'})
    fig1, ax1 = plt.subplots()
    bars = ax1.bar(contagem['risco'], contagem['quantidade'], color=cores)
    ax1.bar_label(bars, padding=3)
    ax1.set_title('Distribuição de Risco de Crédito (sem moderado)')
    ax1.set_xlabel('Classe de Risco')
    ax1.set_ylabel('Nº de Registros')
    fig1.tight_layout()

    # Heatmap de correlação
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    df_num = df.copy()
    df_num['risco_cod'] = y
    sns.heatmap(
        df_num[utils.RiscoCredito.COLUNAS + ['risco_cod']].corr(),
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        linewidths=0.5,
        ax=ax2,
    )
    ax2.set_title('Correlação entre variáveis e risco de crédito')
    fig2.tight_layout()

    plt.show()

# ---------------------------3.7 e 3.8------------------------------- #
def carregar_credit(caminho: str):
    """Carrega o arquivo credit.pkl que contém X_train, y_train, X_test, y_test."""
    with open(caminho, 'rb') as f:
        dados = pickle.load(f)
    X_train, y_train, X_test, y_test = dados[0], dados[1], dados[2], dados[3]
    return X_train, y_train, X_test, y_test


def plotar_matriz_confusao(cm, classes, titulo, ax):
    """Plota uma Matriz de Confusão estilizada no eixo fornecido."""
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')


def parte3_credit_data(caminho: str = 'Parte3/credit.pkl'):
    """
    PARTE 3 (7-8) — Regressão Logística na base Credit Data
    Avalia acurácia, precisão, recall e compara com Naive Bayes e Florestas Aleatórias.
    """
    print("\n" + "=" * 65)
    print("PARTE 3: Regressão Logística — Credit Data")
    print("=" * 65)

    X_train, y_train, X_test, y_test = carregar_credit(caminho)

    modelo_lr = LogisticRegression(random_state=1)
    modelo_lr.fit(X_train, y_train)

    y_pred_lr = modelo_lr.predict(X_test)
    cm_lr     = confusion_matrix(y_test, y_pred_lr)
    acuracia_lr  = accuracy_score(y_test, y_pred_lr)

    print(f"\nAcurácia Regressão Logística: {acuracia_lr * 100:.2f}%")
    
    # ... (restante da lógica de comparação dos colegas)
    return X_train, y_train, X_test, y_test

def parte4_svm_credit_data(X_train, y_train, X_test, y_test):
    """
    PARTE 4: SVM
    """
    print("\n" + "=" * 65)
    print("PARTE 4: SVM")
    print("=" * 65)

    # 2. Inicialmente treine o SVM com kernel linear, valor do parâmetro C = 1.0 e 'random_state =1'
    svm_linear = SVC(kernel='linear', C=1.0, random_state=1)
    svm_linear.fit(X_train, y_train)

    # 3. Utilize o comando do sklearn accuray_score para calcular a acurácia. O resultado deve ser 0.946
    y_pred_linear = svm_linear.predict(X_test)
    acuracia_linear = accuracy_score(y_test, y_pred_linear)
    print(f"Acurácia (Linear): {acuracia_linear:.3f}")

    # 4. Teste os demais kernels e anote os resultados.
    kernels = ['poly', 'sigmoid', 'rbf']
    print("\nTestando outros kernels:")
    for k in kernels:
        svm = SVC(kernel=k, C=1.0, random_state=1)
        svm.fit(X_train, y_train)
        acc = accuracy_score(y_test, svm.predict(X_test))
        print(f"   Kernel {k:7}: {acc:.3f}")

    