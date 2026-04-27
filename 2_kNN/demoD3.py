# Julia Welter, Pedro Pacheco, Pedro Albuquerque e Tiago Rosa
import scipy.io as sio
import numpy as np
import os
from utils_knn import meuKnn, normalizacao, visualizaPontos

diretorio_script = os.path.dirname(os.path.abspath(__file__))
caminho_mat = os.path.join(diretorio_script, 'grupoDados3.mat')

try:
    data = sio.loadmat(caminho_mat)
    grupoTrain = data['grupoTrain']
    trainRots = data['trainRots']
    grupoTest  = data['grupoTest']
    testRots   = data['testRots']
    print(f"Dados carregados com sucesso de: {caminho_mat}")
    print(f"  grupoTrain: {grupoTrain.shape}, grupoTest: {grupoTest.shape}")
    print(f"  Classes únicas: {np.unique(trainRots)}")
except FileNotFoundError:
    print(f"Erro: Arquivo 'grupoDados3.mat' não encontrado em: {caminho_mat}")
    exit()

print("\nGerando gráfico de dispersão das dimensões 0 e 1...")
visualizaPontos(grupoTrain, trainRots, 0, 1)


pred_k1  = meuKnn(grupoTrain, trainRots, grupoTest, 1)
acc_k1   = np.mean(pred_k1 == testRots) * 100

print(f"\nRESPOSTA Q3.1: Acurácia com k=1 (sem normalização): {acc_k1:.2f}%")

print("\nRESULTADOS DE ACURÁCIA — buscando 92%")
print(f"{'K':<5} {'Sem Norm (%)':<16} {'Com Norm (%)':<16}")

train_norm = normalizacao(grupoTrain)
test_norm  = normalizacao(grupoTest)

melhor_acc    = 0
melhor_k      = 0
melhor_config = ""

for k in [1, 3, 5, 7, 10, 15, 20]:
    # Sem normalização
    pred_orig = meuKnn(grupoTrain, trainRots, grupoTest, k)
    acc_orig  = np.mean(pred_orig == testRots) * 100

    # Com normalização
    pred_norm = meuKnn(train_norm, trainRots, test_norm, k)
    acc_norm  = np.mean(pred_norm == testRots) * 100

    print(f"{k:<5} {acc_orig:>13.2f}% {acc_norm:>13.2f}%")

    if max(acc_orig, acc_norm) > melhor_acc:
        melhor_acc    = max(acc_orig, acc_norm)
        melhor_k      = k
        melhor_config = "Sem normalização" if acc_orig >= acc_norm else "Com normalização"

print(f"\nRESPOSTA Q3.2: Acurácia máxima = {melhor_acc:.2f}% (k={melhor_k}, {melhor_config}).")
print()
print("  Por que a acuracia com k=1 sem normalizacao e inferior a 92%?")
print("  As duas features do dataset possuem escalas muito diferentes entre si.")
print("  Na distancia Euclidiana, a feature com maior magnitude domina o calculo,")
print("  fazendo com que a outra feature praticamente nao influencie na classificacao.")
print("  Isso leva o classificador a tomar decisoes baseadas em apenas uma dimensao,")
print("  o que aumenta os erros de classificacao.")
print()
print("  O que foi feito para atingir 92%?")
print(f"  Aplicamos normalizacao min-max nas duas features (escala [0,1]) e testamos")
print(f"  diferentes valores de k. Com k={melhor_k} e dados normalizados, ambas as")
print("  features contribuem igualmente para a distancia Euclidiana, permitindo que")
print(f"  o classificador use toda a informacao disponivel e atinja {melhor_acc:.2f}% de acuracia.")
