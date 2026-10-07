from fase2_utils import carregar_base, montar_X_y


def executar(base=None):
    if base is None:
        base = carregar_base()

    features = base.columns[:-1].tolist()
    target = base.columns[-1]

    x_base, x, y = montar_X_y(base)
    print("c) X e y gerados corretamente.")
    print(f"X_base contem apenas as variaveis independentes: {', '.join(features)}.")
    print(f"X contem a coluna de 1 para o intercepto, seguida de {', '.join(features)}.")
    print(f"y contem a variavel dependente: {target}.")
    print(f"Shape de X_base: {x_base.shape}")
    print(f"Shape de X: {x.shape}")
    print(f"Shape de y: {y.shape}")
    print("\nPrimeiras linhas de X_base:")
    print(base[features].head().to_string(index=False))
    print("\nPrimeiras linhas de X:")
    # Para visualizacao, mostramos o intercepto e as features
    print(base.assign(intercepto=1)[["intercepto"] + features].head().to_string(index=False))
    print("\nPrimeiras linhas de y:")
    print(base[[target]].head().to_string(index=False))
    return x_base, x, y


if __name__ == "__main__":
    executar()
