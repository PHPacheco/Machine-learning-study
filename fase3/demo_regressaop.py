import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipy
import os

# Caminho para os arquivos de dados
file_mat = os.path.join(os.path.dirname(__file__), 'data_preg.mat')
file_csv = os.path.join(os.path.dirname(__file__), 'data_preg.csv')

# a) Carregar o arquivo data_preg (suporta .mat ou .csv)
if os.path.exists(file_mat):
    mat = scipy.loadmat(file_mat)
    # Tenta extrair 'data' conforme snippet do enunciado
    if 'data' in mat:
        data = mat['data']
    else:
        # Se não houver 'data', tenta 'x' e 'y' ou a primeira chave disponível
        keys = [k for k in mat.keys() if not k.startswith('__')]
        data = mat[keys[0]]
    x = data[:, 0]
    y = data[:, 1]
else:
    # Se não houver .mat, usa .csv
    data = pd.read_csv(file_csv, header=None)
    x = data[0].values
    y = data[1].values

# b) Gráfico de dispersão dos dados
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Dados Originais', alpha=0.6)

# polyfit retorna [beta_N, beta_{N-1}, ..., beta_0]
def calcular_polinomio(coeffs, x_vals):
    n = len(coeffs) - 1
    y_vals = np.zeros_like(x_vals, dtype=float)
    for i, beta in enumerate(coeffs):
        y_vals += beta * (x_vals ** (n - i))
    return y_vals

# c) N = 1 (Cor vermelha)
# polyfit: beta0=betaN, beta1=betaN-1...
beta1 = np.polyfit(x, y, 1)
y1 = calcular_polinomio(beta1, x)
plt.plot(x, y1, 'r', label='Regressão N=1')

# d) N = 2 (Cor verde)
beta2 = np.polyfit(x, y, 2)
y2 = calcular_polinomio(beta2, x)
plt.plot(x, y2, 'g', label='Regressão N=2')

# e) N = 3 (Cor preta)
beta3 = np.polyfit(x, y, 3)
y3 = calcular_polinomio(beta3, x)
plt.plot(x, y3, 'k', label='Regressão N=3')

# f) N = 8 (Cor amarela)
beta8 = np.polyfit(x, y, 8)
y8 = calcular_polinomio(beta8, x)
plt.plot(x, y8, 'y', label='Regressão N=8')

# Configurações do gráfico
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regressão Polinomial - Fase 3')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Mostrar o gráfico
print("Exibindo o gráfico de regressão polinomial...")
plt.show()
