"""
prever_temperatura.py

Estima temp_media das próximas 1 a 4 semanas pela climatologia: média da
mesma semana do ano nos anos anteriores (+-1 semana). Compara com o
baseline "repetir a semana atual" no período de teste (últimos 20%) e
gera SQL com temp_prevista, temp_min e temp_max.

Como rodar (da pasta raiz do ml-dengue):
    python src/prever_temperatura.py
"""

import numpy as np
from datetime import timedelta
from pathlib import Path

from data_prep import carregar_dados, HORIZONTES_SEMANAS

SAIDA = Path(__file__).resolve().parent.parent / "data" / "processed" / "temp_previsoes.sql"


def semana_do_ano(datas):
    return datas.dt.isocalendar().week.astype(int)


def climatologia(historico, semana_alvo):
    """Temperaturas do histórico na mesma semana do ano (+-1 semana)."""
    semanas = semana_do_ano(historico["data_inicio"])
    dist = np.minimum(np.abs(semanas - semana_alvo), 52 - np.abs(semanas - semana_alvo))
    return historico.loc[dist <= 1, "temp_media"].dropna().values


def main():
    df = carregar_dados().sort_values("data_inicio").reset_index(drop=True)
    corte = int(len(df) * 0.8)
    treino = df.iloc[:corte]

    # --- teste: climatologia (só com dados do treino) vs repetir semana atual ---
    print("Erro médio da temperatura (graus):")
    for n in HORIZONTES_SEMANAS:
        erros_clim, erros_base = [], []
        for i in range(corte, len(df) - n):
            alvo = df.iloc[i + n]
            sem = int(alvo["data_inicio"].isocalendar().week)
            clim = climatologia(treino, sem)
            if len(clim) == 0:
                continue
            erros_clim.append(abs(alvo["temp_media"] - clim.mean()))
            erros_base.append(abs(alvo["temp_media"] - df.iloc[i]["temp_media"]))
        print(f"  +{n} sem  repetir atual: {np.mean(erros_base):.2f} | "
              f"climatologia: {np.mean(erros_clim):.2f}")

    # --- previsão real, a partir da última semana do banco ---
    ultima = df.iloc[-1]
    data_base = ultima["data_inicio"].date()
    print(f"\nSemana base: {data_base} | temp_media nessa semana: {ultima['temp_media']:.1f}")

    sql = [
        "ALTER TABLE previsoes ADD COLUMN IF NOT EXISTS temp_prevista NUMERIC(4,1);",
        "ALTER TABLE previsoes ADD COLUMN IF NOT EXISTS temp_min NUMERIC(4,1);",
        "ALTER TABLE previsoes ADD COLUMN IF NOT EXISTS temp_max NUMERIC(4,1);",
    ]
    for n in HORIZONTES_SEMANAS:
        alvo = data_base + timedelta(weeks=n)
        sem = alvo.isocalendar()[1]
        clim = climatologia(df, sem)
        prev = float(np.mean(clim))
        p10, p90 = np.percentile(clim, [10, 90])
        print(f"+{n} sem ({alvo}): {prev:.1f} graus (faixa {p10:.1f} a {p90:.1f})")
        sql.append(
            f"UPDATE previsoes SET temp_prevista = {prev:.1f}, temp_min = {p10:.1f}, "
            f"temp_max = {p90:.1f} WHERE data_base = '{data_base}' AND horizonte_semanas = {n};"
        )

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text("\n".join(sql) + "\n", encoding="utf-8")
    print(f"\nSQL salvo em: {SAIDA}")


if __name__ == "__main__":
    main()