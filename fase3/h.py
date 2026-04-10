from fase3_utils import carregar_dados, dividir_treino_teste



class QuestaoH:
    def executar(self):
        x, y = carregar_dados()
        # test_size=0.1 para 10% de teste
        x_treino, x_teste, y_treino, y_teste = dividir_treino_teste(x, y, test_size=0.1)
        
        print(f"--- Questão H ---")
        print(f"Dados totais: {len(x)}")
        print(f"Dados de treinamento: {len(x_treino)}")
        print(f"Dados de teste: {len(x_teste)}")
        
        return x_treino, x_teste, y_treino, y_teste

if __name__ == "__main__":
    QuestaoH().executar()