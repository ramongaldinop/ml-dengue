"""
prever_casos.py

Estima o número de casos de dengue nas próximas 1 a 4 semanas (a partir da
última semana do banco) e gera um SQL que acrescenta as colunas
casos_previstos, casos_min e casos_max na tabela previsoes.

A faixa (min/max) vem da variação entre as árvores do Random Forest
(percentis 10 e 90). É uma faixa de variação do modelo, não uma garantia.

Como rodar (da pasta raiz do ml-dengue):
    python src/prever_casos.py
"""

import joblib
import numpy as np
from datetime import timedelta
from pathlib import Path

from data_prep import preparar_dataset, HORIZONTES_SEMANAS
from train_casos import FEATURES

RAIZ = Path(__file__).resolve().parent.parent
MODELS_DIR = RAIZ / "models"
SAIDA = RAIZ / "data" / "processed" / "casos_previsoes.sql"


def main():
    _df, linha, _features, _targets = preparar_dataset()

    data_base = linha["data_inicio"].iloc[0].date()
    casos_log_hoje = float(linha["casos_log"].iloc[0])
    X = linha[FEATURES]

    print(f"Semana base: {data_base} | casos nessa semana: {np.expm1(casos_log_hoje):.0f}")

    sql = [
        "ALTER TABLE previsoes ADD COLUMN IF NOT EXISTS casos_previstos INTEGER;",
        "ALTER TABLE previsoes ADD COLUMN IF NOT EXISTS casos_min INTEGER;",
        "ALTER TABLE previsoes ADD COLUMN IF NOT EXISTS casos_max INTEGER;",
    ]

    for n in HORIZONTES_SEMANAS:
        modelo = joblib.load(MODELS_DIR / f"rf_casos_{n}sem.joblib")

        delta = float(modelo.predict(X)[0])
        deltas_arvores = np.array([arv.predict(X.values)[0] for arv in modelo.estimators_])
        p10, p90 = np.percentile(deltas_arvores, [10, 90])

        previsto = int(round(np.expm1(casos_log_hoje + delta)))
        minimo = int(round(np.expm1(casos_log_hoje + p10)))
        maximo = int(round(np.expm1(casos_log_hoje + p90)))
        minimo, maximo = min(minimo, previsto), max(maximo, previsto)

        alvo = data_base + timedelta(weeks=n)
        print(f"+{n} sem ({alvo}): {previsto} casos (faixa {minimo} a {maximo})")

        sql.append(
            f"UPDATE previsoes SET casos_previstos = {previsto}, "
            f"casos_min = {minimo}, casos_max = {maximo} "
            f"WHERE data_base = '{data_base}' AND horizonte_semanas = {n};"
        )

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text("\n".join(sql) + "\n", encoding="utf-8")
    print(f"\nSQL salvo em: {SAIDA}")


if __name__ == "__main__":
    main()