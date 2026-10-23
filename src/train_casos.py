"""
train_casos.py

Treina Random Forest (regressão) pra estimar o NÚMERO de casos de dengue
daqui a 1, 2, 3 e 4 semanas. Em vez de prever o número direto, prevê a
VARIAÇÃO em relação à semana atual (em log) e depois converte pra casos.

Como rodar (da pasta raiz do ml-dengue):
    python src/train_casos.py
"""

import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from data_prep import (
    carregar_dados,
    criar_features_defasadas,
    criar_features_crescimento,
    HORIZONTES_SEMANAS,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURES = [
    "temp_media", "umid_media", "rt", "mes", "nivel_alerta",
    "casos_log", "cresc_2sem", "cresc_4sem",
    "temp_media_lag2", "umid_media_lag2",
    "temp_media_lag4", "umid_media_lag4",
]


def preparar_dataset_casos():
    """Mesma preparação do data_prep.py, mas com alvos de casos em vez de nível."""
    df = carregar_dados()
    df["rt"] = df["rt"].fillna(0)
    df = criar_features_defasadas(df)
    df = criar_features_crescimento(df)

    for n in HORIZONTES_SEMANAS:
        # alvo = quanto o log dos casos muda daqui a n semanas
        df[f"delta_casos_{n}sem"] = df["casos_log"].shift(-n) - df["casos_log"]
        # casos reais daqui a n semanas (só pra comparar no teste)
        df[f"casos_{n}sem"] = df["casos"].shift(-n)

    alvos = [f"delta_casos_{n}sem" for n in HORIZONTES_SEMANAS]
    df_treino = df.dropna(subset=FEATURES + alvos).reset_index(drop=True)
    return df_treino


def main():
    df = preparar_dataset_casos().sort_values("data_inicio").reset_index(drop=True)

    corte = int(len(df) * 0.8)
    treino = df.iloc[:corte]
    teste = df.iloc[corte:]

    print(f"Treino: {treino['data_inicio'].min().date()} até "
          f"{treino['data_inicio'].max().date()} ({len(treino)} semanas)")
    print(f"Teste:  {teste['data_inicio'].min().date()} até "
          f"{teste['data_inicio'].max().date()} ({len(teste)} semanas)")

    MODELS_DIR.mkdir(exist_ok=True)

    for n in HORIZONTES_SEMANAS:
        alvo = f"delta_casos_{n}sem"
        print("\n" + "=" * 60)
        print(f"HORIZONTE: {n} semana(s) a frente")
        print("=" * 60)

        modelo = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        modelo.fit(treino[FEATURES], treino[alvo])

        # previsão da variação -> converte pra número de casos
        delta_pred = modelo.predict(teste[FEATURES])
        casos_prev = np.expm1(teste["casos_log"].values + delta_pred)
        casos_real = teste[f"casos_{n}sem"].values
        casos_base = teste["casos"].values  # baseline: repete a semana atual

        erro_rf = np.abs(casos_real - casos_prev)
        erro_base = np.abs(casos_real - casos_base)

        print(f"Erro médio (casos de diferença):")
        print(f"  Baseline (repetir semana atual): {erro_base.mean():.1f}")
        print(f"  Random Forest:                   {erro_rf.mean():.1f}")
        print(f"Erro mediano:")
        print(f"  Baseline: {np.median(erro_base):.1f} | Random Forest: {np.median(erro_rf):.1f}")

        caminho = MODELS_DIR / f"rf_casos_{n}sem.joblib"
        joblib.dump(modelo, caminho)
        print(f"Modelo salvo em: {caminho}")


if __name__ == "__main__":
    main()