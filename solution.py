import numpy as np
import pandas as pd


def solution(arquivo):
    """Retorna [intercepto, coeficiente_area, coeficiente_quartos]."""
    df = pd.read_csv(arquivo)

    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    y = np.array(df["preco_venda"].to_list(), dtype=float)

    # Cada linha de X representa [1, area, quartos].
    # A coluna de uns permite estimar o intercepto; id_imovel nao entra.
    matriz = []
    for i in range(len(area)):
        matriz.append([1, area[i], quartos[i]])
    X = np.array(matriz)

    # Formula matricial apresentada no slide de Regressao linear:
    # beta = inversa(X transposta * X) * X transposta * y
    transposta = np.transpose(X)
    produto = np.dot(transposta, X)
    inversa = np.linalg.inv(produto)
    beta = np.dot(np.dot(inversa, transposta), y)
    return beta.tolist()


if __name__ == "__main__":
    import sys

    arquivo = sys.argv[1] if len(sys.argv) > 1 else "imoveis_lar_perfeito.csv"
    coeficientes = solution(arquivo)
    print(coeficientes)
    b0, b1, b2 = coeficientes
    print(f"Preco = {b0:.6f} + {b1:.6f} * Area + {b2:.6f} * Quartos")
