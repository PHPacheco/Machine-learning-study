import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px


# ============================================================
# PARTE 2 - Base de Dados: Credit Data (Grande)
# ============================================================
# Nesta seção testamos a Árvore de Decisão para a Base de Dados
# Credit Risk Dataset. O objetivo é classificar clientes que
# pagam (classe 0) ou não pagam a dívida (classe 1), a fim de
# o banco decidir sobre a concessão de empréstimo.

# a) Abrir o arquivo e verificar o tamanho dos dados
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, \
    X_credit_teste, y_credit_teste = pickle.load(f)

# Verificação das dimensões
# Esperado: X_treinamento=(1500, 3), y_treinamento=(1500,)
#           X_teste=(500, 3),        y_teste=(500,)
print("\n--- Dimensões dos dados (Credit Data) ---")
print(f"X treinamento : {X_credit_treinamento.shape}")   # (1500, 3)
print(f"y treinamento : {y_credit_treinamento.shape}")   # (1500,)
print(f"X teste       : {X_credit_teste.shape}")         # (500, 3)
print(f"y teste       : {y_credit_teste.shape}")         # (500,)

# b) Treinar o modelo de Árvore de Decisão
# random_state=0 garante reprodutibilidade dos resultados
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)

# c) Previsão com os dados de teste
previsoes_credit = arvore_credit.predict(X_credit_teste)

# Visualização: comparação entre previsões e valores reais
print("\n--- Comparação previsões x valores reais (primeiros 20) ---")
comparacao = pd.DataFrame({
    'Previsão' : previsoes_credit[:20],
    'Real'     : y_credit_teste[:20]
})
print(comparacao.to_string(index=False))

# d) Cálculo da acurácia e relatório de classificação
from sklearn.metrics import accuracy_score, classification_report

acuracia = accuracy_score(y_credit_teste, previsoes_credit)
print(f"\nAcurácia do modelo (Credit Data): {acuracia:.4f} ({acuracia*100:.2f}%)")

print("\nRelatório de Classificação:")
print(classification_report(
    y_credit_teste,
    previsoes_credit,
    target_names=['Paga (0)', 'Não paga (1)']
))


# o CODIGO abaixo nao mexer


# Previsões
previsoes = arvore_credit.predict(X_credit_teste)

# ─────────────────────────────────────────────
# e) MATRIZ DE CONFUSÃO
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("e) MATRIZ DE CONFUSÃO")
print("=" * 55)

mat = confusion_matrix(y_credit_teste, previsoes)
vp = mat[0][0]   # Pagantes classificados corretamente
fp = mat[0][1]   # Pagantes classificados como não pagantes (erro)
fn = mat[1][0]   # Não pagantes classificados como pagantes (erro)
vn = mat[1][1]   # Não pagantes classificados corretamente

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=mat,
    display_labels=['Pagante (0)', 'Não Pagante (1)']
)
disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
ax_cm.set_title('Matriz de Confusão — Credit Risk', fontsize=13,
                fontweight='bold', pad=12)
fig_cm.tight_layout()
fig_cm.savefig('matriz_confusao.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n  i.   Pagantes classificados CORRETAMENTE (VP):              {vp}")
print(f"  ii.  Pagantes classificados como NÃO PAGANTES - erro (FP): {fp}")
print(f"  iii. Não pagantes classificados CORRETAMENTE (VN):          {vn}")
print(f"  iv.  Não pagantes classificados como PAGANTES - erro (FN):  {fn}")

# ─────────────────────────────────────────────
# f) CLASSIFICATION REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("f) CLASSIFICATION REPORT")
print("=" * 55)

print(classification_report(
    y_credit_teste,
    previsoes,
    target_names=["Pagante (0)", "Não Pagante (1)"]
))

print("""
INTERPRETAÇÃO — Precision vs Recall:

  • Precision (Precisão): dos clientes que o modelo marcou
    como pertencentes a uma classe, quantos realmente são?
    → "Quando o modelo alerta 'não vai pagar', quão
       confiável é esse alerta?"

  • Recall (Sensibilidade): de todos os clientes que
    realmente pertencem a uma classe, quantos o modelo
    detectou?
    → "De todos os inadimplentes reais, quantos o modelo
       conseguiu identificar?"

  CLASSE 0 — Pagante:
    Precision 0.99 e Recall 0.99 → o modelo é excelente
    em identificar bons pagadores. Esperado, pois esta
    classe é maioria nos dados (436 de 500 no teste).

  CLASSE 1 — Não Pagante:
    Precision 0.91 e Recall 0.94 → de 64 inadimplentes
    reais, o modelo detectou ~94% (apenas 4 escaparam).
    Dos alertas emitidos, 91% eram inadimplentes de fato.

  TRADE-OFF no contexto bancário:
    Recall da classe 1 é a métrica mais crítica: um
    inadimplente não detectado (falso negativo) representa
    um empréstimo concedido a quem não pagará — prejuízo
    direto. Neste modelo o recall de 94% é satisfatório.

  Acurácia geral: 98% → 490 de 500 clientes classificados
  corretamente (apenas 10 erros: 6 FP + 4 FN).
""")

# ─────────────────────────────────────────────
# g) VISUALIZAÇÃO DA ÁRVORE DE DECISÃO
# ─────────────────────────────────────────────
print("=" * 55)
print("g) VISUALIZAÇÃO DA ÁRVORE DE DECISÃO")
print("=" * 55)

fig_tree, ax_tree = plt.subplots(figsize=(30, 12))

plot_tree(
    arvore_credit,
    feature_names=['income', 'age', 'loan'],
    class_names=['Pagante', 'Não Pagante'],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax_tree
)

ax_tree.set_title(
    'Árvore de Decisão — Credit Risk Dataset',
    fontsize=16, fontweight='bold', pad=20
)

fig_tree.tight_layout()
fig_tree.savefig('arvore_credit.png', dpi=150, bbox_inches='tight')
print("Árvore salva em: arvore_credit.png")
plt.show()
