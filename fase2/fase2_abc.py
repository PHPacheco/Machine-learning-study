from pathlib import Path

import numpy as np
import pandas as pd


COLUNAS = ["tamanho", "quartos", "preco"]


def carregar_base() -> pd.DataFrame:
    csv_path = Path("fase2/data.csv")
    mat_path = Path("fase2/data.mat")

    if csv_path.exists():
        dados = pd.read_csv(csv_path, header=None, names=COLUNAS)
        return dados

    if mat_path.exists():
        try:
            from scipy.io import loadmat  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "data.mat encontrado, mas scipy nao esta instalado para leitura do .mat."
            ) from exc

        mat = loadmat(mat_path)
        for valor in mat.values():
            if isinstance(valor, np.ndarray) and valor.ndim == 2 and valor.shape[1] >= 3:
                dados = pd.DataFrame(valor[:, :3], columns=COLUNAS)
                return dados

    raise FileNotFoundError("Nao encontrei data.csv nem data.mat na pasta atual.")


def main() -> None:
    base = carregar_base()

    print("a) Base carregada com sucesso.")
    print(f"Total de exemplos: {len(base)}")

    print("\nb) Analise estatistica (.describe()):")
    print(base.describe())

    media_preco = base["preco"].mean()
    indice_menor_casa = base["tamanho"].idxmin()
    preco_menor_casa = base.loc[indice_menor_casa, "preco"]
    indice_mais_cara = base["preco"].idxmax()
    quartos_mais_cara = int(base.loc[indice_mais_cara, "quartos"])

    print("\nRespostas da letra b:")
    print(f"- Media de preco das casas: {media_preco:.2f}")
    print(f"- Quanto custa a menor casa (menor tamanho): {preco_menor_casa:.2f}")
    print(f"- Quantos quartos tem a casa mais cara: {quartos_mais_cara}")

    X = base[["tamanho", "quartos"]].to_numpy(dtype=float)
    y = base["preco"].to_numpy(dtype=float).reshape(-1, 1)

    print("\nc) X e y gerados:")
    print(f"- Shape de X: {X.shape}")
    print(f"- Shape de y: {y.shape}")
    print("\nMatriz X (primeiras 10 linhas):")
    print(base[["tamanho", "quartos"]].head(10).to_string(index=False))
    print("\nVetor y (primeiras 10 linhas):")
    print(base[["preco"]].head(10).to_string(index=False))



if __name__ == "__main__":
    main()
