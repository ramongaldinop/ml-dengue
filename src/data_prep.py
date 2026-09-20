"""
data_prep.py
------------
Prepara o dataset pra treino do modelo de previsao, a partir de um
export CSV da tabela casos_dengue (a mesma tabela que o Murilo populou
via docker-compose em database/).

Por que ler de um CSV exportado, e nao conectar direto no Postgres:
o driver Python (psycopg/psycopg2) no Windows apresentou um bug de
encoding especifico dessa maquina/ambiente, que quebrava a conexao de
rede mesmo com o banco configurado corretamente. Exportar os dados
uma vez via "docker exec" (que ja sabemos que funciona) e ler o CSV
localmente contorna esse problema sem depender de rede nenhuma.

Como gerar/atualizar o export (rodar de dentro da pasta database/,
sempre que o banco for repopulado com dados novos), colando o comando
abaixo SEM as aspas externas no terminal:

    docker exec -t dengue_db psql -U dengue -d dengue_db -c "COPY_CMD" > casos_dengue_export.csv

    onde COPY_CMD e:  copy casos_dengue TO STDOUT WITH CSV HEADER (com uma barra invertida antes de "copy", omitida aqui so pra nao confundir o Python)

Depois e so mover/copiar esse arquivo pra data/raw/ do ml-dengue.

Ao inves de tentar prever usando clima FUTURO (que nao existe ainda),
o dataset e montado assim:
    features  -> temp_media, umid_media, rt, mes da semana ATUAL
    target    -> nivel_alerta de 2 e 4 semanas A FRENTE

Isso deixa o modelo pronto pra, dada a ultima semana real do banco,
prever o nivel de alerta esperado daqui a 2 ou 4 semanas.
"""

from pathlib import Path

import pandas as pd
import numpy as np

# --- Caminho do CSV exportado do banco do Murilo ---
CAMINHO_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "casos_dengue_export.csv"

HORIZONTES_SEMANAS = [2, 4]  # quantas semanas a frente vamos tentar prever


def carregar_dados(caminho_csv=CAMINHO_CSV):
    """Le o CSV exportado do banco, ja ordenado por data.

    O comando de export do PowerShell (docker exec ... > arquivo.csv) salva
    o arquivo em UTF-16 por padrao em algumas versoes do Windows, entao
    tentamos UTF-8 primeiro (padrao) e caimos pra UTF-16 se falhar.
    """
    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Nao encontrei {caminho_csv}.\n"
            "Gere o export primeiro (dentro da pasta database/ do previsao-dengue):\n"
            '  docker exec -t dengue_db psql -U dengue -d dengue_db '
            '-c "\\copy casos_dengue TO STDOUT WITH CSV HEADER" > casos_dengue_export.csv\n'
            "E copie o arquivo gerado pra data/raw/ do ml-dengue."
        )

    try:
        df = pd.read_csv(caminho_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(caminho_csv, encoding="utf-16")

    df["data_inicio"] = pd.to_datetime(df["data_inicio"])
    df = df.sort_values("data_inicio").reset_index(drop=True)
    df["mes"] = df["data_inicio"].dt.month
    return df

# Quantas semanas pra trás olhar pra medir a tendência
LAGS_SEMANAS = [2, 4]


def criar_targets_defasados(df, horizontes=HORIZONTES_SEMANAS):
    """
    Cria uma coluna nivel_alerta_Nsem pra cada horizonte, deslocando o
    nivel_alerta N linhas pra tras (shift negativo = olhar pro futuro).
    Como os dados sao semanais e sem gaps, deslocar N linhas equivale a
    deslocar N semanas.
    """
    df = df.copy()
    for n in horizontes:
        df[f"nivel_alerta_{n}sem"] = df["nivel_alerta"].shift(-n)
    return df


def criar_features_defasadas(df, lags=LAGS_SEMANAS):
    """
    Cria variaveis com valores de semanas ANTERIORES (shift positivo =
    olhar pro passado). Isso da ao modelo nocao de tendencia: nao so como
    esta hoje, mas se esta subindo ou caindo, e o clima de semanas atras.
    """
    df = df.copy()
    for n in lags:
        for coluna in ["nivel_alerta", "casos", "temp_media", "umid_media"]:
            df[f"{coluna}_lag{n}"] = df[coluna].shift(n)
    return df

def criar_features_crescimento(df):
    """
    Cria variaveis de crescimento dos casos. Usamos log porque os casos
    vao de poucas centenas a milhares: log da diferenca = "quantas vezes
    os casos multiplicaram" nas ultimas 2 e 4 semanas.
    """
    df = df.copy()
    df["casos_log"] = np.log1p(df["casos"])
    df["cresc_2sem"] = df["casos_log"] - np.log1p(df["casos_lag2"])
    df["cresc_4sem"] = df["casos_log"] - np.log1p(df["casos_lag4"])
    return df

def preparar_dataset():
    """Pipeline completo: carrega, cria features e targets, separa treino/previsao."""
    df = carregar_dados()
    df["rt"] = df["rt"].fillna(0)
    df = criar_features_defasadas(df)
    df = criar_features_crescimento(df)
    df = criar_targets_defasados(df)

    features = [
        "temp_media", "umid_media", "rt", "mes", "nivel_alerta",
        "casos_log", "cresc_2sem", "cresc_4sem",
        "temp_media_lag2", "umid_media_lag2",
        "temp_media_lag4", "umid_media_lag4",
    ]
    
    targets = [f"nivel_alerta_{n}sem" for n in HORIZONTES_SEMANAS]

    # TREINO: precisa ter features e target conhecidos. As primeiras semanas
    # ficam sem "passado" (lags) e as ultimas sem "futuro" (target).
    df_treino = df.dropna(subset=features + targets).reset_index(drop=True)

    # Linha mais recente do banco = ponto de partida pra prever o futuro de verdade
    linha_mais_recente = df.iloc[[-1]][["data_inicio"] + features]

    return df_treino, linha_mais_recente, features, targets


if __name__ == "__main__":
    total_no_banco = len(carregar_dados())
    df_treino, linha_mais_recente, features, targets = preparar_dataset()

    print(f"Total de linhas carregadas do CSV: {total_no_banco}")
    print(f"Linhas utilizaveis pro treino: {len(df_treino)}")
    print(f"Linhas descartadas (sem passado ou sem futuro): {total_no_banco - len(df_treino)}")
    print(f"Semanas com rt == 0: {(df_treino['rt'] == 0).sum()}")
    print(f"\nFeatures usadas ({len(features)}): {features}")
    print(f"Targets criados: {targets}\n")

    print("Amostra do dataset de treino:")
    print(df_treino[["data_inicio"] + features + targets].head())

    print("\nUltima semana disponivel no banco (usada pra prever o futuro real):")
    print(linha_mais_recente)