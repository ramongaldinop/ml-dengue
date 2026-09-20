# ml-dengue

Modelo de machine learning do projeto de monitoramento e previsão de dengue em São Paulo (capital), feito no PJI410 da UNIVESP.

O modelo é um Random Forest que prevê o **nível de alerta** da dengue daqui a **2 e 4 semanas**, usando dados de casos, clima e o índice rt.

## Como rodar

1. Crie e ative o ambiente virtual e instale as dependências:
```
   pip install -r requirements.txt
```
2. Coloque o CSV de dados em `data/raw/` (a pasta `data/` não vai pro GitHub).
3. Rode os scripts nesta ordem:
```
   python src/data_prep.py   # limpa os dados e cria as variáveis
   python src/train.py       # treina os modelos de 2 e 4 semanas
   python src/avaliar.py     # avalia o modelo ano a ano
   python src/prever.py      # gera as previsões
```

## Integração com o banco

O `prever.py` gera um arquivo SQL com as previsões, que é carregado na tabela `previsoes` do Postgres (Docker). O backend em Node só lê essa tabela.

## Estrutura

- `src/`: scripts em Python
- `data/`: dados (fora do GitHub)
- `requirements.txt`: dependências