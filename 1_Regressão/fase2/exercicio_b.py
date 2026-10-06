from fase2_utils import carregar_base


def executar(base=None):
    if base is None:
        base = carregar_base()

    features = base.columns[:-1].tolist()
    target = base.columns[-1]

    describe = base.describe()
    
    print("b) Analise estatistica (.describe()):")
    print(describe)

    if target in base.columns:
        media_alvo = base[target].mean()
        print(f"\nMedia de {target} das casas: {media_alvo:.2f}")
        
        indice_mais_cara = base[target].idxmax()
        print(f"Index do exemplo com maior {target}: {indice_mais_cara}")
        
    # Tenta mostrar informacoes baseadas nos nomes esperados, se existirem
    if "tamanho" in base.columns and target in base.columns:
        indice_menor = base["tamanho"].idxmin()
        valor_alvo_menor = base.loc[indice_menor, target]
        print(f"Preco (ou alvo) da menor casa (pelo tamanho): {valor_alvo_menor:.2f}")

    if "quartos" in base.columns and target in base.columns:
        indice_cara = base[target].idxmax()
        qtd_quartos = int(base.loc[indice_cara, "quartos"])
        print(f"Quartos da casa mais cara: {qtd_quartos}")

    return describe


if __name__ == "__main__":
    executar()
