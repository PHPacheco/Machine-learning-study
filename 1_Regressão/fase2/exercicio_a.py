from Regressão.fase2.fase2_utils import carregar_base


def executar():
    base = carregar_base()
    print("a) Base carregada com sucesso.")
    print(f"Total de exemplos: {len(base)}")
    print(base.head().to_string(index=False))
    return base


if __name__ == "__main__":
    executar()
