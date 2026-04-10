import numpy as np

from fase2_utils import calcular_theta, carregar_base


def executar(base=None):
    if base is None:
        base = carregar_base()

    features = base.columns[:-1].tolist()
    target = base.columns[-1]
    num_features = len(features)
    
    theta = calcular_theta(base).flatten()
    
    # Valores base para a previsao (tamanho 1650 e quartos 2, 3, 4)
    # Se nao houver as colunas esperadas, usamos as duas primeiras features.
    val_feat1 = 1650
    vals_feat2 = [3, 2, 4]
    
    # Para features extras (se houver), usamos a media
    medias = base[features].mean().to_numpy()
    
    def predizer(f2_val):
        x_pred = np.ones(num_features + 1)
        x_pred[1] = val_feat1
        x_pred[2] = f2_val
        if num_features > 2:
            x_pred[3:] = medias[2:]
        return float(x_pred @ theta)

    previsoes = [predizer(v) for v in vals_feat2]

    print(f"h) Previsao para {features[0]} = {val_feat1} e variando {features[1]}:")
    print("Coeficientes usados na previsao:")
    print(f"beta0 (intercepto): {theta[0]:.4f}")
    for i, col in enumerate(features):
        print(f"beta{i+1} ({col}): {theta[i+1]:.6f}")
    
    print()
    for v, p in zip(vals_feat2, previsoes):
        print(f"{v} {features[1]}: {p:.2f}")
   
    if num_features >= 2:
        print(
            f"\nNota: Ao variar {features[1]}, observamos o impacto direto do coeficiente beta2 "
            f"no {target} previsto, mantendo as outras variaveis fixas."
        )
    
    return tuple(previsoes)


if __name__ == "__main__":
    executar()
