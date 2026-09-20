"""
prever.py

Treina o Random Forest com TODAS as semanas disponiveis e prevê o nivel
de alerta de 2 e 4 semanas a frente, a partir da ultima semana real do banco.
Gera um arquivo SQL (data/processed/previsoes.sql) que cria a tabela
`previsoes` e grava as previsoes nela.

Como rodar (da pasta raiz do ml-dengue):
    python src/prever.py
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_prep import HORIZONTES_SEMANAS, preparar_dataset

SAIDA_SQL = Path(__file__).resolve().parent.parent / "data" / "processed" / "previsoes.sql"

CRIAR_TABELA = """
CREATE TABLE IF NOT EXISTS previsoes (
    id SERIAL PRIMARY KEY,
    data_base DATE NOT NULL,              -- ultima semana real usada pra prever
    horizonte_semanas INTEGER NOT NULL,   -- 2 ou 4 semanas a frente
    data_alvo DATE NOT NULL,              -- data_base + horizonte
    nivel_previsto INTEGER NOT NULL CHECK (nivel_previsto BETWEEN 1 AND 4),
    probabilidade NUMERIC(4,3),           -- parte das arvores que votou nesse nivel
    criado_em TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (data_base, horizonte_semanas)
);
"""


def main():
    df, linha_mais_recente, features, targets = preparar_dataset()
    data_base = linha_mais_recente["data_inicio"].iloc[0]
    X_novo = linha_mais_recente[features]

    print(f"Ultima semana real no banco: {data_base.date()}\n")

    comandos = [CRIAR_TABELA]

    for n, alvo in zip(HORIZONTES_SEMANAS, targets):
        modelo = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        modelo.fit(df[features], df[alvo].astype(int))

        nivel = int(modelo.predict(X_novo)[0])
        probabilidade = float(modelo.predict_proba(X_novo)[0].max())
        data_alvo = data_base + pd.Timedelta(weeks=n)

        print(f"{n} semanas a frente ({data_alvo.date()}): "
              f"nivel {nivel} (probabilidade {probabilidade:.0%})")

        comandos.append(
            "INSERT INTO previsoes "
            "(data_base, horizonte_semanas, data_alvo, nivel_previsto, probabilidade)\n"
            f"VALUES ('{data_base.date()}', {n}, '{data_alvo.date()}', {nivel}, {probabilidade:.3f})\n"
            "ON CONFLICT (data_base, horizonte_semanas) DO UPDATE SET\n"
            "    data_alvo = EXCLUDED.data_alvo,\n"
            "    nivel_previsto = EXCLUDED.nivel_previsto,\n"
            "    probabilidade = EXCLUDED.probabilidade,\n"
            "    criado_em = now();\n"
        )

    SAIDA_SQL.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_SQL.write_text("\n".join(comandos), encoding="utf-8")
    print(f"\nArquivo SQL gerado: {SAIDA_SQL}")


if __name__ == "__main__":
    main()