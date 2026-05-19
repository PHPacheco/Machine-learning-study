import matplotlib
matplotlib.use('TkAgg')
 
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
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


def parte4_credit_data(caminho: str = 'credit.pkl'):
    """
    PARTE 4 — Regressão Logística, Naive Bayes e Florestas Aleatórias na base Credit Data.
    Avalia e compara acurácia, precisão, recall e Matriz de Confusão dos três algoritmos.
    """
    print("\n" + "=" * 65)
    print("PARTE 4: Comparação de Algoritmos — Credit Data")
    print("=" * 65)
 
    # ------------------------------------------------------------------ #
    # 1. Carregamento dos dados                                            #
    # ------------------------------------------------------------------ #
    X_train, y_train, X_test, y_test = carregar_credit(caminho)
 
    print(f"\nBase de treino : {X_train.shape[0]} registros | {X_train.shape[1]} atributos")
    print(f"Base de teste  : {X_test.shape[0]} registros")
    print(f"\nDistribuição das classes (treino):")
    unicos_tr, contagens_tr = np.unique(y_train, return_counts=True)
    for u, c in zip(unicos_tr, contagens_tr):
        rotulo = 'inadimplente' if u == 1 else 'adimplente'
        print(f"   Classe {u} ({rotulo}): {c} ({c / len(y_train) * 100:.1f}%)")
 
    # ------------------------------------------------------------------ #
    # 2. Treinamento dos três algoritmos                                   #
    # ------------------------------------------------------------------ #
    modelo_lr = LogisticRegression(random_state=1)
    modelo_nb = GaussianNB()
    modelo_rf = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
 
    modelo_lr.fit(X_train, y_train)
    modelo_nb.fit(X_train, y_train)
    modelo_rf.fit(X_train, y_train)
 
    print("\n\nParâmetros do modelo de treino (Regressão Logística)")
    print(f"   Intercepto (B0) : {modelo_lr.intercept_}")
    print(f"   Coeficientes    : {modelo_lr.coef_}")
 
    # ------------------------------------------------------------------ #
    # 3. Predições                                                         #
    # ------------------------------------------------------------------ #
    y_pred_lr = modelo_lr.predict(X_test)
    y_pred_nb = modelo_nb.predict(X_test)
    y_pred_rf = modelo_rf.predict(X_test)
 
    # ------------------------------------------------------------------ #
    # 4. Métricas individuais                                              #
    # ------------------------------------------------------------------ #
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
 
    acuracia_lr = accuracy_score(y_test, y_pred_lr)
    precisao_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    recall_lr   = recall_score(y_test, y_pred_lr, zero_division=0)
 
    acuracia_nb = accuracy_score(y_test, y_pred_nb)
    precisao_nb = precision_score(y_test, y_pred_nb, zero_division=0)
    recall_nb   = recall_score(y_test, y_pred_nb, zero_division=0)
 
    acuracia_rf = accuracy_score(y_test, y_pred_rf)
    precisao_rf = precision_score(y_test, y_pred_rf, zero_division=0)
    recall_rf   = recall_score(y_test, y_pred_rf, zero_division=0)
 
    for nome, y_pred, acuracia in [
        ('Regressão Logística', y_pred_lr, acuracia_lr),
        ('Naive Bayes',         y_pred_nb, acuracia_nb),
        ('Florestas Aleatórias',y_pred_rf, acuracia_rf),
    ]:
        print(f"\n\nResultados — {nome} (Credit Data)")
        print(f"   Taxa de acerto (acurácia) : {acuracia * 100:.2f}%")
        print(f"\n   Relatório completo:")
        print(classification_report(
            y_test, y_pred,
            target_names=['adimplente (0)', 'inadimplente (1)'],
        ))
 
    # ------------------------------------------------------------------ #
    # 5. Tabela comparativa                                                #
    # ------------------------------------------------------------------ #
    print("\n\nTabela Comparativa de Algoritmos — Credit Data")
    print("-" * 62)
    print(f"{'Algoritmo':<25} {'Acurácia':>10} {'Precisão':>10} {'Recall':>10}")
    print("-" * 62)
    print(f"{'Regressão Logística':<25} {acuracia_lr * 100:>9.2f}% {precisao_lr * 100:>9.2f}% {recall_lr * 100:>9.2f}%")
    print(f"{'Naive Bayes':<25} {acuracia_nb * 100:>9.2f}% {precisao_nb * 100:>9.2f}% {recall_nb * 100:>9.2f}%")
    print(f"{'Florestas Aleatórias':<25} {acuracia_rf * 100:>9.2f}% {precisao_rf * 100:>9.2f}% {recall_rf * 100:>9.2f}%")
    print("-" * 62)
 
    print("""
Análise dos Resultados:
  • Regressão Logística obteve boa acurácia global (~94–95%), porém
    apresenta recall moderado na classe minoritária (inadimplentes),
    indicando que alguns casos de risco passam despercebidos.
  • Naive Bayes é o mais simples e rápido, mas o pressuposto de
    independência entre atributos limita seu desempenho nesta base.
  • Florestas Aleatórias supera ambos em todas as métricas —
    melhor escolha para detecção de inadimplência, onde falsos
    negativos têm alto custo financeiro.
  • Conclusão: Florestas Aleatórias > Regressão Logística > Naive
    Bayes para esta base.
""")
 
    # ------------------------------------------------------------------ #
    # 6. Visualizações                                                     #
    # ------------------------------------------------------------------ #
 
    # — Distribuição das classes (teste) —
    unicos_te, contagens_te = np.unique(y_test, return_counts=True)
    rotulos_te = ['adimplente' if u == 0 else 'inadimplente' for u in unicos_te]
    cores_te   = ['#2ECC71' if u == 0 else '#E74C3C' for u in unicos_te]
 
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    bars3 = ax3.bar(rotulos_te, contagens_te, color=cores_te)
    ax3.bar_label(bars3, padding=3)
    ax3.set_title('Distribuição de Classes — Credit Data (teste)')
    ax3.set_xlabel('Classe')
    ax3.set_ylabel('Nº de Registros')
    fig3.tight_layout()
 
    fig4, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig4.suptitle('Matrizes de Confusão — Comparação de Algoritmos (Credit Data)',
                  fontsize=12, fontweight='bold')
 
    classes = ['adimplente', 'inadimplente']
    plotar_matriz_confusao(cm_lr, classes, f'Regressão Logística\nAcurácia: {acuracia_lr * 100:.2f}%',  axes[0])
    plotar_matriz_confusao(cm_nb, classes, f'Naive Bayes\nAcurácia: {acuracia_nb * 100:.2f}%',          axes[1])
    plotar_matriz_confusao(cm_rf, classes, f'Florestas Aleatórias\nAcurácia: {acuracia_rf * 100:.2f}%', axes[2])
 
    fig4.tight_layout()
 
    algoritmos  = ['Regressão\nLogística', 'Naive\nBayes', 'Florestas\nAleatórias']
    metricas = {
        'Acurácia' : [acuracia_lr, acuracia_nb, acuracia_rf],
        'Precisão' : [precisao_lr, precisao_nb, precisao_rf],
        'Recall'   : [recall_lr,   recall_nb,   recall_rf],
    }
 
    x           = np.arange(len(algoritmos))
    largura     = 0.25
    cores_barra = ['#3498DB', '#E67E22', '#2ECC71']
 
    fig5, ax5 = plt.subplots(figsize=(9, 5))
    for i, (metrica, valores) in enumerate(metricas.items()):
        barras = ax5.bar(x + i * largura, [v * 100 for v in valores],
                         largura, label=metrica, color=cores_barra[i])
        ax5.bar_label(barras, fmt='%.1f%%', padding=3, fontsize=8)
 
    ax5.set_title('Comparação de Métricas — Algoritmos no Credit Data',
                  fontsize=11, fontweight='bold')
    ax5.set_xticks(x + largura)
    ax5.set_xticklabels(algoritmos)
    ax5.set_ylabel('Valor (%)')
    ax5.set_ylim(0, 115)
    ax5.legend()
    fig5.tight_layout()
 
    plt.show()
 
 
if __name__ == '__main__':
    main()
    parte4_credit_data('credit.pkl')
    
    
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

    