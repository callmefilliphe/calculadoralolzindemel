# LoL Live Betting — Calculadora de Probabilidade (Polymarket)

Metodologia própria para apostas ao vivo (moneyline) em partidas profissionais de
League of Legends via Polymarket. O modelo compara uma probabilidade de vitória
estimada (treinada com dados históricos da Oracle's Elixir) contra o preço do
contrato na Polymarket, calculando o edge e o tamanho de aposta sugerido (Kelly).

## Como funciona

1. Modelos de regressão logística foram treinados com ~24 mil partidas profissionais
   (2024-2026, dados da [Oracle's Elixir](https://oracleselixir.com/tools/downloads)),
   em 4 cortes de tempo (10, 15, 20 e 25 minutos de jogo).
2. As variáveis usadas são só as que aparecem direto na transmissão ao vivo:
   **diferença de gold**, **diferença de kills** e **lado do mapa** (blue/red).
3. Durante uma partida ao vivo, você lê esses 3 valores na tela, roda a calculadora,
   e compara a probabilidade do modelo com o preço atual do contrato na Polymarket.

## Desempenho dos modelos (AUC / Brier score no conjunto de teste)

| Corte de tempo | AUC | Brier |
|---|---|---|
| 10 min | 0.759 | 0.199 |
| 15 min | 0.818 | 0.174 |
| 20 min | 0.875 | 0.144 |
| 25 min | 0.917 | 0.116 |

AUC mais próximo de 1.0 = melhor discriminação. Brier mais baixo = melhor calibração.

## Instalação

Requer Python 3.8+.

```bash
pip install -r requirements.txt
```

## Uso

1. Abra `calculadora_live.py`
2. Edite o bloco `SEUS DADOS AQUI` com os valores da partida em andamento:
   - `minuto`: minuto atual do jogo
   - `golddiff`: diferença de gold do seu time (lido da barra da transmissão)
   - `killdiff`: kills do seu time menos kills do adversário
   - `is_blue`: 1 se seu time está no lado azul, 0 se vermelho
   - `preco_polymarket`: preço atual do contrato (ex: 0.62 = 62 centavos)
3. Rode:

```bash
python3 calculadora_live.py
```

Saída esperada:

```
Probabilidade do modelo:  82.5%
Preco da Polymarket:      62.0%
Edge:                     +20.5 pontos
Stake sugerido (teto 5%): 5.00% do bankroll
```

## Regras de decisão

- **Edge mínimo para entrar:** 8 pontos percentuais (margem de segurança contra
  erro do modelo, taxas e spread da Polymarket)
- **Stake:** Kelly fracionado (40% do Kelly cheio), com teto manual de 5% do
  bankroll por aposta — protege contra excesso de confiança do modelo em
  cenários extremos (ex: gold diffs muito altos, raros no treino)
- **Evitar entrar logo após eventos óbvios** (ace, abate de Baron) — o mercado
  já reprecifica rápido nesse instante

## Próximos passos possíveis

- Log de apostas (planilha) para medir CLV (closing line value) e Brier score
  real ao longo do tempo, validando se o modelo realmente gera edge
- Reincorporar xp diff / cs diff caso a transmissão mostre o scoreboard completo
- Automatizar a leitura do preço da Polymarket via API (gamma/CLOB)

## Arquivos

- `calculadora_live.py` — script principal, edite e rode para cada consulta
- `models_simple.pkl` — modelos treinados (coeficientes por corte de tempo)
- `train_simple.py` — script de treino original (caso queira retreinar com dados novos)
