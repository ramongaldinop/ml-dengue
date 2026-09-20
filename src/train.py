"""
train.py

Treina um Random Forest pra prever o nível de alerta de dengue
daqui a 2 semanas e daqui a 4 semanas (um modelo pra cada).
Divide treino/teste por data: treina no passado, testa no futuro.

Como rodar (da pasta raiz do ml-dengue):
    python src/train.py
"""

import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data_prep import preparar_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LABELS = [1, 2, 3, 4]


def main():
    df, _linha_mais_recente, features, targets = preparar_dataset()

    df = df.sort_values("data_inicio").reset_index(drop=True)

    # primeiros 80% das semanas = treino, últimos 20% = teste
    corte = int(len(df) * 0.8)
    treino = df.iloc[:corte]
    teste = df.iloc[corte:]

    print(f"Treino: {treino['data_inicio'].min().date()} até "
          f"{treino['data_inicio'].max().date()} ({len(treino)} semanas)")
    print(f"Teste:  {teste['data_inicio'].min().date()} até "
          f"{teste['data_inicio'].max().date()} ({len(teste)} semanas)")

    # quantas semanas de cada nível existem no treino e no teste
    for alvo in targets:
        print(f"Distribuição de {alvo} no treino: "
              f"{treino[alvo].astype(int).value_counts().sort_index().to_dict()}")
        print(f"Distribuição de {alvo} no teste:  "
              f"{teste[alvo].astype(int).value_counts().sort_index().to_dict()}")

    MODELS_DIR.mkdir(exist_ok=True)

    for alvo in targets:
        print("\n" + "=" * 60)
        print(f"ALVO: {alvo}")
        print("=" * 60)

        X_train, y_train = treino[features], treino[alvo].astype(int)
        X_test, y_test = teste[features], teste[alvo].astype(int)

        # baseline: "daqui a X semanas vai estar igual a hoje"
        y_base = teste["nivel_alerta"].astype(int)

        modelo = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        print(f"Baseline (repetir nível atual): "
              f"acurácia {accuracy_score(y_test, y_base):.3f} | "
              f"F1 macro {f1_score(y_test, y_base, average='macro', zero_division=0):.3f}")
        print(f"Random Forest:                  "
              f"acurácia {accuracy_score(y_test, y_pred):.3f} | "
              f"F1 macro {f1_score(y_test, y_pred, average='macro', zero_division=0):.3f}")

        print("\n--- Relatório de classificação ---")
        print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))
        print("--- Matriz de confusão (linhas = real, colunas = previsto) ---")
        print(confusion_matrix(y_test, y_pred, labels=LABELS))

        print("\n--- Importância de cada variável ---")
        for nome, imp in zip(features, modelo.feature_importances_):
            print(f"{nome}: {imp:.3f}")

        caminho = MODELS_DIR / f"rf_{alvo}.joblib"
        joblib.dump(modelo, caminho)
        print(f"Modelo salvo em: {caminho}")


if __name__ == "__main__":
    main()