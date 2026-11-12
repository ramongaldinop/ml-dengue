# ml-dengue

Modelo de machine learning do projeto de monitoramento e previsão de dengue em São Paulo (capital), feito no PJI410 da UNIVESP (projeto D-Vision).

O modelo é um Random Forest que prevê o nível de alerta da dengue daqui a **1, 2, 3 e 4 semanas**, usando dados de casos, clima e o índice rt. Também estima o número de casos e a temperatura para essas semanas.

## Como rodar

1. Crie e ative o ambiente virtual e instale as dependências:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Gere o CSV a partir do banco. Com o Docker rodando, abra o terminal na pasta `database/` do projeto `previsao-dengue` e rode:

```
docker exec -t dengue_db psql -U dengue -d dengue_db -c "\copy casos_dengue TO STDOUT WITH CSV HEADER" > casos_dengue_export.csv
```

Depois copie o arquivo `casos_dengue_export.csv` para `data/raw/` do `ml-dengue` (a pasta `data/` não vai pro GitHub, então crie ela se não existir).

3. Rode os scripts nesta ordem:

```
python src/data_prep.py          # limpa os dados e cria as variáveis
python src/train.py              # treina os modelos de nível de alerta (1 a 4 semanas)
python src/avaliar.py            # avalia o nível de alerta ano a ano
python src/train_casos.py        # treina os modelos de estimativa de casos
python src/prever.py             # gera as previsões de nível de alerta (previsoes.sql)
python src/prever_casos.py       # estima casos, com faixa mínima e máxima (casos_previsoes.sql)
python src/prever_temperatura.py # estima a temperatura (temp_previsoes.sql)
```

> Se o Python do seu Windows bloquear o scipy (Controle de Aplicativo), rode os scripts via Docker com a imagem `python:3.12`, montando a pasta do projeto.

4. Carregue as previsões no banco. Com o Docker rodando, abra o terminal na pasta raiz do `ml-dengue` e rode:

```
docker cp data/processed/previsoes.sql dengue_db:/tmp/previsoes.sql
docker exec -t dengue_db psql -U dengue -d dengue_db -f /tmp/previsoes.sql

docker cp data/processed/casos_previsoes.sql dengue_db:/tmp/casos_previsoes.sql
docker exec -t dengue_db psql -U dengue -d dengue_db -f /tmp/casos_previsoes.sql

docker cp data/processed/temp_previsoes.sql dengue_db:/tmp/temp_previsoes.sql
docker exec -t dengue_db psql -U dengue -d dengue_db -f /tmp/temp_previsoes.sql
```

O `previsoes.sql` cria a tabela `previsoes` (se ainda não existir) e grava o nível de alerta previsto. Os outros dois acrescentam as colunas de casos e de temperatura e preenchem as linhas que já existem, por isso **precisam rodar depois** do `previsoes.sql`. Para conferir:

```
docker exec -t dengue_db psql -U dengue -d dengue_db -c "SELECT * FROM previsoes;"
```

Rodar de novo com uma semana nova atualiza as previsões já existentes, sem duplicar.

## Como funciona

Em vez de usar previsão do tempo (que não existe ainda), o modelo usa a situação da semana atual e das semanas anteriores para prever o futuro.

**Nível de alerta**
- Entradas (12 variáveis): temperatura e umidade médias (atuais e de 2 e 4 semanas atrás), rt, mês, nível de alerta atual, casos (em log) e crescimento dos casos em 2 e 4 semanas.
- Saída: nível de alerta (1 a 4) daqui a 1, 2, 3 e 4 semanas, cada horizonte com um modelo próprio.

**Casos e temperatura (estimativas)**
- Casos: Random Forest de regressão que prevê a variação do log de casos. A faixa mínima/máxima vem dos percentis 10 e 90 das árvores.
- Temperatura: climatologia, ou seja, a média da mesma semana do ano nos anos anteriores.

O treino e a avaliação respeitam a ordem do tempo: o modelo aprende com o passado e é testado no futuro (divisão 80/20 por data).

## Resultados

Em acurácia, o Random Forest fica praticamente empatado com o baseline "repetir o nível atual". Na avaliação ano a ano (2015–2025), o baseline ganha em 1, 2 e 3 semanas e o modelo só ganha em 4 semanas. Em 2024 o nível ficou em 4 o ano todo e o modelo não captou a queda.

Nas estimativas, o modelo de casos tem erro médio menor que o baseline nos quatro horizontes (o baseline tem erro mediano menor), e a climatologia de temperatura ganhou do baseline nos quatro horizontes.

## Integração com o banco

Os scripts de previsão geram três arquivos SQL em `data/processed/`: `previsoes.sql` (cria a tabela `previsoes` no Postgres e grava o nível de alerta previsto), `casos_previsoes.sql` (colunas `casos_previstos`, `casos_min`, `casos_max`) e `temp_previsoes.sql` (colunas `temp_prevista`, `temp_min`, `temp_max`). Os dois últimos usam `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` e `UPDATE`, então podem ser rodados mais de uma vez. O backend em Node só lê essa tabela, sem chamar o Python diretamente. O ML não roda em produção: é executado manualmente e o resultado é carregado no banco.

## Estrutura

- `src/`: scripts em Python
  - `data_prep.py`: carrega o CSV e cria as variáveis e os alvos (1 a 4 semanas)
  - `train.py`: treina os modelos de nível de alerta e salva em `models/`
  - `avaliar.py`: avalia o modelo ano a ano, comparando com o baseline de repetir o nível atual
  - `train_casos.py`: treina o modelo de estimativa de casos
  - `prever.py`: gera as previsões de nível de alerta e o SQL para o banco
  - `prever_casos.py`: estima casos com faixa mínima e máxima
  - `prever_temperatura.py`: estima a temperatura por climatologia
  - `diagnostico_2024.py`: mostra semana a semana o que o modelo previu em 2024
- `data/`: dados (fora do GitHub)
- `models/`: modelos treinados (fora do GitHub)
- `requirements.txt`: dependências
