"""
avaliar.py

Avaliação mais confiável que o train.py: em vez de UMA divisão
treino/teste, testa ano por ano. Pra cada ano, treina só com os anos
ANTERIORES e testa naquele ano (como seria na vida real).

Como rodar (da pasta raiz do ml-dengue):
    python src/avaliar.py
"""

from sklearn.ensemble import RandomForestClassifier

from data_prep import preparar_dataset

PRIMEIRO_ANO_TESTE = 2015  # 2010-2014 ficam só pra treino


def main():
    df, _linha_mais_recente, features, targets = preparar_dataset()
    df["ano"] = df["data_inicio"].dt.year

    for alvo in targets:
        print("\n" + "=" * 70)
        print(f"ALVO: {alvo}")
        print("=" * 70)
        print("ano  | semanas | acerto baseline | acerto RF | nível 4 perdido (base / RF)")

        total = ok_base_total = ok_rf_total = 0

        for ano in sorted(df["ano"].unique()):
            if ano < PRIMEIRO_ANO_TESTE:
                continue

            treino = df[df["ano"] < ano]
            teste = df[df["ano"] == ano]

            modelo = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            modelo.fit(treino[features], treino[alvo].astype(int))

            real = teste[alvo].astype(int)
            pred = modelo.predict(teste[features])
            base = teste["nivel_alerta"].astype(int)

            ok_base = int((base == real).sum())
            ok_rf = int((pred == real).sum())
            # "perdido" = era nível 4 (vermelho) e o palpite foi 1 ou 2
            perdido_base = int(((real == 4) & (base <= 2)).sum())
            perdido_rf = int(((real == 4) & (pred <= 2)).sum())

            n = len(teste)
            print(f"{ano} | {n:7d} | {ok_base / n:15.3f} | {ok_rf / n:9.3f} | "
                  f"{perdido_base} / {perdido_rf}")

            total += n
            ok_base_total += ok_base
            ok_rf_total += ok_rf

        print(f"TOTAL | {total:6d} | {ok_base_total / total:15.3f} | "
              f"{ok_rf_total / total:9.3f}")


if __name__ == "__main__":
    main()