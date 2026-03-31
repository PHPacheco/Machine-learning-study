from fase2_utils import carregar_base


def executar(base=None):
    if base is None:
        base = carregar_base()

    describe = base.describe()
    media_preco = base["preco"].mean()
    indice_menor_casa = base["tamanho"].idxmin()
    preco_menor_casa = base.loc[indice_menor_casa, "preco"]
    indice_mais_cara = base["preco"].idxmax()
    quartos_mais_cara = int(base.loc[indice_mais_cara, "quartos"])

    print("b) Analise estatistica (.describe()):")
    print(describe)
    print(f"\nMedia de preco das casas: {media_preco:.2f}")
    print(f"Preco da menor casa: {preco_menor_casa:.2f}")
    print(f"Quartos da casa mais cara: {quartos_mais_cara}")
    return describe


if __name__ == "__main__":
    executar()
