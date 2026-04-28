# Julia Welter, Pedro Pacheco, Pedro Albuquerque e Tiago Rosa
import numpy as np
import matplotlib.pyplot as plt

def dist(p, q):
   return np.sqrt(np.sum((p - q)**2))

def normalizacao(dados):
    min_val = np.min(dados, axis=0)
    max_val = np.max(dados, axis=0)
   
    dados_norm = (dados - min_val) / (max_val - min_val + 1e-10)
    return dados_norm

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    num_teste = dadosTeste.shape[0]
    num_treino = dadosTrain.shape[0]
    previsoes = []

    for i in range(num_teste):
        distancias = []
        for j in range(num_treino):
            d = dist(dadosTeste[i, :], dadosTrain[j, :])
            
            distancias.append((d, rotuloTrain[j][0]))

        distancias.sort(key=lambda x: x[0])

        k_vizinhos = [distancias[n][1] for n in range(k)]

        rotulos_unicos, contagens = np.unique(k_vizinhos, return_counts=True)
        voto_vencedor = rotulos_unicos[np.argmax(contagens)]
        
        previsoes.append(voto_vencedor)
        
    return np.array(previsoes).reshape(-1, 1)

def visualizaPontos(dados, rotulos, d1, d2):
    plt.figure(figsize=(10, 6))
    classes = np.unique(rotulos)
    cores = ['red', 'blue', 'green', 'orange', 'purple']
    marcadores = ['^', '+', '.', 'x', 'o']
    
    for i, c in enumerate(classes):
        indices = np.where(rotulos == c)[0]
        plt.scatter(dados[indices, d1], dados[indices, d2], 
                    c=cores[i % len(cores)], 
                    marker=marcadores[i % len(marcadores)], 
                    label=f'Classe {int(c)}')
    
    plt.xlabel(f'Atributo (Dimensão {d1})')
    plt.ylabel(f'Atributo (Dimensão {d2})')
    plt.title(f'Visualização de Dados: Dimensões {d1} vs {d2}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
