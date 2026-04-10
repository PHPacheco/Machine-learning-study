# i.py
import matplotlib.pyplot as plt
import numpy as np
from c import calcular_polinomio
from fase3.h import QuestaoH

class QuestaoI:
    def executar(self, exibir=True):
        # Obtém os dados divididos da Questão H
        x_treino, x_teste, y_treino, y_teste = QuestaoH().executar()
        
        # Criamos o gráfico de dispersão com os dois conjuntos
        plt.figure(figsize=(10, 6))
        plt.scatter(x_treino, y_treino, color="blue", label="Treino (90%)", alpha=0.5)
        plt.scatter(x_teste, y_teste, color="orange", label="Teste (10%)", marker='x')

      
        graus = [1, 2, 3, 8]
        cores = {1: 'r', 2: 'g', 3: 'k', 8: 'y'}
        
      
        x_min = min(x_treino.min(), x_teste.min())
        x_max = max(x_treino.max(), x_teste.max())
        x_suave = np.linspace(x_min, x_max, 500)

        print("Ajustando modelos apenas com dados de treinamento...")

        for n in graus:
          
            beta = np.polyfit(x_treino, y_treino, n)
            
          
            y_curva = calcular_polinomio(beta, x_suave)
            
           
            plt.plot(x_suave, y_curva, cores[n], label=f"Regressão N={n} (Treino)")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Questão I: Ajuste Polinomial apenas com Dados de Treino")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        if exibir:
            plt.show()
            
        return x_treino, x_teste, y_treino, y_teste

if __name__ == "__main__":
    QuestaoI().executar(exibir=True)