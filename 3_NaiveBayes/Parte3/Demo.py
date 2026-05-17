import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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

    print(f"\Encoder - Classes codificadas: {dict(enumerate(label_encoder.classes_))}")
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

if __name__ == '__main__':
    main()