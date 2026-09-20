"""
diagnostico_2024.py

Mostra, semana a semana, o que o modelo de 4 semanas previu em 2024
(treinando só com os anos anteriores), pra entender onde ele errou.

Como rodar (da pasta raiz do ml-dengue):
    python src/diagnostico_2024.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_prep import preparar_dataset

ANO = 2024
ALVO = "nivel_alerta_4sem"


def main():
    df, _linha_mais_recente, features, _targets = preparar_dataset()
    df["ano"] = df["data_inicio"].dt.year

    treino = df[df["ano"] < ANO]
    teste = df[df["ano"] == ANO].copy()

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    modelo.fit(treino[features], treino[ALVO].astype(int))

    teste["real_daqui_4sem"] = teste[ALVO].astype(int)
    teste["previsto"] = modelo.predict(teste[features])

    pd.set_option("display.width", 200)
    colunas = ["data_inicio", "nivel_alerta", "casos", "real_daqui_4sem", "previsto"]
    print(teste[colunas].to_string(index=False))

    erros = (teste["real_daqui_4sem"] != teste["previsto"]).sum()
    print(f"\nErros: {erros} de {len(teste)} semanas")
    print(f"Maior número de casos no treino (antes de {ANO}): {int(treino['casos'].max())}")
    print(f"Maior número de casos em {ANO}: {int(teste['casos'].max())}")


if __name__ == "__main__":
    main()