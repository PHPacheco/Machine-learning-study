from fase2_utils import carregar_base, montar_X_y


def executar(base=None):
    if base is None:
        base = carregar_base()

    x_base, x, y = montar_X_y(base)
    print("c) X e y gerados corretamente.")
    print("X_base contem apenas as variaveis independentes: tamanho e quartos.")
    print("X contem a coluna de 1 para o intercepto, seguida de tamanho e quartos.")
    print("y contem a variavel dependente: preco.")
    print(f"Shape de X_base: {x_base.shape}")
    print(f"Shape de X: {x.shape}")
    print(f"Shape de y: {y.shape}")
    print("\nPrimeiras linhas de X_base:")
    print(base[["tamanho", "quartos"]].head().to_string(index=False))
    print("\nPrimeiras linhas de X:")
    print(base.assign(intercepto=1)[["intercepto", "tamanho", "quartos"]].head().to_string(index=False))
    print("\nPrimeiras linhas de y:")
    print(base[["preco"]].head().to_string(index=False))
    return x_base, x, y


if __name__ == "__main__":
    executar()
