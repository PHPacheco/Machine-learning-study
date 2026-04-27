# Julia Welter, Pedro Pacheco, Pedro Albuquerque e Tiago Rosa
import scipy.io as sio
import numpy as np
import os
from utils_knn import meuKnn, normalizacao, visualizaPontos

diretorio_script = os.path.dirname(os.path.abspath(__file__))
caminho_mat = os.path.join(diretorio_script, 'grupoDados2.mat')

try:
    dados = sio.loadmat(caminho_mat)
    grupoTrain = dados['grupoTrain']
    trainRots = dados['trainRots']
    grupoTest = dados['grupoTest']
    testRots = dados['testRots']
    print(f"Dados carregados com sucesso de: {caminho_mat}")
except FileNotFoundError:
    print(f"Erro: Arquivo 'grupoDados2.mat' não encontrado em: {caminho_mat}")
    exit()

caracteristicas = [
    "Álcool", "Ácido málico", "Cinzas", "Alcalinidade das cinzas",
    "Magnésio", "Fenóis totais", "Flavonóides", "Fenóis não flavonóides",
    "Proantocianinas", "Intensidade de cor", "Tonalidade",
    "OD280/OD315", "Prolina"
]

# Conjunto de treino: 118 amostras, 13 características
# Conjunto de teste:  60 amostras

# Exibe escala das features para evidenciar o problema de normalização
print("\nCaracterísticas                    :    mín   /    máx")
for i, nome in enumerate(caracteristicas):
    print(f"{nome:<35}: {grupoTrain[:, i].min():>8.2f} / {grupoTrain[:, i].max():>8.2f}")

# gráfico de dispersão: Álcool vs Prolina (dim 0 e 12)
visualizaPontos(grupoTrain, trainRots, 0, 12)

# -------------------------------------------------------------------
# Diagnóstico: por que a acurácia sem normalização é baixa?
#
# O kNN calcula distâncias euclidianas entre amostras. Quando as
# características têm escalas muito diferentes, as de maior escala
# dominam o cálculo. Neste dataset a Prolina (feature 12) varia de
# ~278 a ~1547, enquanto Fenóis não flavonóides (feature 7) varia de
# ~0.13 a ~0.63, uma diferença de 1 unidade em Prolina já vale a
# toda a variação de Fenóis, tornando as demais features
# quase irrelevantes para o cálculo de distância.

# A normalização Min-Max escala todas as features para [0,1],
# garantindo que cada uma contribua igualmente.
# -------------------------------------------------------------------

print("\nResultados de acurácia")
print(f"{'K':<5} {'Sem normalização (%)':<22} {'Com normalização (%)':<22}")

melhor_acc = 0
melhor_k = 0
melhor_config = ""

valores_k = [1, 3, 5, 7, 10]

# Normalização aplicada ao conjunto completo antes do loop
trainNorm = normalizacao(grupoTrain)
testNorm = normalizacao(grupoTest)

for k in valores_k:
    # Sem normalização
    pred_orig = meuKnn(grupoTrain, trainRots, grupoTest, k)
    acc_orig = np.mean(pred_orig == testRots) * 100

    # Com normalização Min-Max
    pred_norm = meuKnn(trainNorm, trainRots, testNorm, k)
    acc_norm = np.mean(pred_norm == testRots) * 100

    print(f"{k} {acc_orig:>9.2f}% {acc_norm:>9.2f}%")

    if max(acc_orig, acc_norm) > melhor_acc:
        melhor_acc = max(acc_orig, acc_norm)
        melhor_k = k
        melhor_config = "Sem normalização" if acc_orig >= acc_norm else "Com normalização"

print(f"\nMelhor resultado: acurácia = {melhor_acc:.2f}% (k={melhor_k}, {melhor_config})\n\n")

# -------------------------------------------------------------------
# Conclusão:
# Sem normalização, a acurácia fica entre ~60% e ~78% dependendo de 
# features de grande escala (ex: Prolina ≈ 278-1547), prolina e o 
# Magnésio dominam o espaço.

# Com normalização Min-Max,todas as features contribuem igualmente, 
# e a acurácia supera 92% para qualquer k testado.