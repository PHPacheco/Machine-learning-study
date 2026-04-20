import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

diretorio_script = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(diretorio_script, 'dataset_risco_credito.csv')

try:
    dataset_risco_credito = pd.read_csv(caminho_csv)
    print(f"Dataset carregado com sucesso de: {caminho_csv}")
    print("\nPrimeiras linhas do dataset:")
    print(dataset_risco_credito.head())
except FileNotFoundError:
    print(f"Erro: Arquivo dataset_risco_credito.csv não encontrado em: {caminho_csv}")
    exit()

# a) DIVISÃO DA BASE DE DADOS
X_risco_credito = dataset_risco_credito.iloc[:, 0:4].values
y_risco_credito = dataset_risco_credito.iloc[:, 4].values

print("\nExemplo de X (antes do LabelEncoder):")
print(X_risco_credito[0])

# b) APLICAR LABEL ENCODER
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:, 0] = label_encoder_historia.fit_transform(X_risco_credito[:, 0])
X_risco_credito[:, 1] = label_encoder_divida.fit_transform(X_risco_credito[:, 1])
X_risco_credito[:, 2] = label_encoder_garantia.fit_transform(X_risco_credito[:, 2])
X_risco_credito[:, 3] = label_encoder_renda.fit_transform(X_risco_credito[:, 3])

print("\nExemplo de X (depois do LabelEncoder):")
print(X_risco_credito[0])

# c) SALVAR O ARQUIVO PRÉ-PROCESSADO
with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)

print("\nArquivo 'risco_credito.pkl' salvo.")
