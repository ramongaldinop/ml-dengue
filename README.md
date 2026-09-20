# ml-dengue

Modelo de machine learning do projeto de monitoramento e previsão de dengue em São Paulo (capital), feito no PJI410 da UNIVESP.

O modelo é um Random Forest que prevê o **nível de alerta** da dengue daqui a **2 e 4 semanas**, usando dados de casos, clima e o índice rt.

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
   python src/data_prep.py   # limpa os dados e cria as variáveis
   python src/train.py       # treina os modelos de 2 e 4 semanas
   python src/avaliar.py     # avalia o modelo ano a ano
   python src/prever.py      # gera as previsões
```

## Como funciona

Em vez de usar previsão do tempo (que não existe ainda), o modelo usa a situação da semana atual e das semanas anteriores para prever o futuro:

- **Entradas:** temperatura e umidade médias (atuais e de 2 e 4 semanas atrás), rt, mês, nível de alerta atual, casos e crescimento dos casos em 2 e 4 semanas.
- **Saída:** nível de alerta (1 a 4) daqui a 2 semanas e daqui a 4 semanas, cada um com um modelo próprio.

O treino e a avaliação respeitam a ordem do tempo: o modelo aprende com o passado e é testado no futuro.

## Integração com o banco

O `prever.py` gera um arquivo SQL (`data/processed/previsoes.sql`) que cria a tabela `previsoes` no Postgres (Docker) e grava as previsões nela. O backend em Node só lê essa tabela, sem chamar o Python diretamente.

## Estrutura

- `src/`: scripts em Python
  - `data_prep.py`: carrega o CSV e cria as variáveis e os alvos
  - `train.py`: treina os modelos e salva em `models/`
  - `avaliar.py`: avalia o modelo ano a ano, comparando com o baseline de repetir o nível atual
  - `prever.py`: gera as previsões e o SQL para o banco
  - `diagnostico_2024.py`: mostra semana a semana o que o modelo previu em 2024
- `data/`: dados (fora do GitHub)
- `models/`: modelos treinados (fora do GitHub)
- `requirements.txt`: dependências