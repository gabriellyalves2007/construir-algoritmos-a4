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
    X = np.column_stack((np.ones(len(df)), area, quartos))

    # Minimos quadrados: minimizar ||X @ beta - y||^2.
    # Equacoes normais: (X.T @ X) @ beta = X.T @ y.
    # lstsq resolve o ajuste sem calcular explicitamente uma inversa.
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta.tolist()


if __name__ == "__main__":
    import sys

    arquivo = sys.argv[1] if len(sys.argv) > 1 else "imoveis_lar_perfeito.csv"
    coeficientes = solution(arquivo)
    print(coeficientes)
    b0, b1, b2 = coeficientes
    print(f"Preco = {b0:.6f} + {b1:.6f} * Area + {b2:.6f} * Quartos")
