# Julia Welter, Pedro Pacheco, Pedro Albuquerque e Tiago Rosa
import scipy.io as sio
import numpy as np
import os
from utils_knn import meuKnn, normalizacao, visualizaPontos

diretorio_script = os.path.dirname(os.path.abspath(__file__))
caminho_mat = os.path.join(diretorio_script, 'grupoDados1.mat')

try:
    data = sio.loadmat(caminho_mat)
    grupoTrain = data['grupoTrain']
    trainRots = data['trainRots']
    grupoTest = data['grupoTest']
    testRots = data['testRots']
    print(f"Dados carregados com sucesso de: {caminho_mat}")
except FileNotFoundError:
    print(f"Erro: Arquivo 'grupoDados1.mat' não encontrado em: {caminho_mat}")
    exit()

print("Gerando gráfico de dispersão das dimensões 2 e 3...")
visualizaPontos(grupoTrain, trainRots, 2, 3)

print("\nRESULTADOS DE ACURÁCIA")
print(f"{'K':<5} {'Original (%)':<15} {'Normalizado (%)':<15}")

melhor_acc = 0
melhor_k = 0
melhor_config = ""

valores_k = [1, 3, 5, 7, 10]

for k in valores_k:
    # Caso 1: SEM NORMALIZAÇÃO
    pred_orig = meuKnn(grupoTrain, trainRots, grupoTest, k)
    acc_orig = np.mean(pred_orig == testRots) * 100
    
    # Caso 2: COM NORMALIZAÇÃO
    train_norm = normalizacao(grupoTrain)
    test_norm = normalizacao(grupoTest)
    pred_norm = meuKnn(train_norm, trainRots, test_norm, k)
    acc_norm = np.mean(pred_norm == testRots) * 100
    
    print(f"{k:<5} {acc_orig:>12.2f}% {acc_norm:>12.2f}%")
    
    # Atualiza a melhor acurácia global
    if max(acc_orig, acc_norm) > melhor_acc:
        melhor_acc = max(acc_orig, acc_norm)
        melhor_k = k
        melhor_config = "Original" if acc_orig >= acc_norm else "Normalizado"

print(f"\nRESPOSTA Q1.1: A acurácia máxima conseguida foi {melhor_acc:.2f}% (k={melhor_k}, {melhor_config}).")
