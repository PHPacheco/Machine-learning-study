# m.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from c import calcular_polinomio
from fase3.h import QuestaoH

def calcular_mse(y_real, y_prev):
    return np.mean((y_real - y_prev) ** 2)

class QuestaoM:
    def executar(self, exibir=True):
        
        x_tr, x_te, y_tr, y_te = QuestaoH().executar()
        
        graus = [1, 2, 3, 8, 20]
        cores = {1: 'r', 2: 'g', 3: 'k', 8: 'y', 20: 'm'} 
        resultados = []

        plt.figure(figsize=(12, 7))
        plt.scatter(x_tr, y_tr, color="blue", alpha=0.4, label="Treino")
        plt.scatter(x_te, y_te, color="orange", label="Teste", marker='x')

        x_suave = np.linspace(x_tr.min(), x_tr.max(), 1000)

        print(f"{'Grau':<6} | {'MSE Treino':<12} | {'MSE Teste':<12}")
        print("-" * 40)

        for n in graus:
          
            beta = np.polyfit(x_tr, y_tr, n)
            
         
            y_pred_tr = calcular_polinomio(beta, x_tr)
            y_pred_te = calcular_polinomio(beta, x_te)
            
          
            mse_tr = calcular_mse(y_tr, y_pred_tr)
            mse_te = calcular_mse(y_te, y_pred_te)
            
            resultados.append({"grau": n, "mse_treino": mse_tr, "mse_teste": mse_te})
            print(f"{n:<6} | {mse_tr:<12.5f} | {mse_te:<12.5f}")

          
            y_plot = calcular_polinomio(beta, x_suave)
            plt.plot(x_suave, y_plot, cores[n], label=f"N={n}")

        plt.ylim(y_tr.min() - 2, y_tr.max() + 2) 
        plt.title("Comparação de Modelos e Overfitting (N=20)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if exibir:
            plt.show()

if __name__ == "__main__":
    QuestaoM().executar()