"""
Calculadora de probabilidade ao vivo (LoL) - versao pratica.
So precisa de 3 numeros que estao sempre visiveis na transmissao:
  - minuto do jogo
  - gold diff (barra superior)
  - diferenca de kills (placar)

Uso:
    python3 calculadora_live.py
(edite os valores no bloco "SEUS DADOS AQUI" antes de rodar)
"""
import pickle
import numpy as np

with open('/home/claude/models_simple.pkl', 'rb') as f:
    MODELS = pickle.load(f)

TIME_CUTS = sorted(MODELS.keys())


def _nearest_cut(minuto):
    validos = [t for t in TIME_CUTS if t <= minuto]
    return max(validos) if validos else min(TIME_CUTS)


def prob_vitoria(minuto, golddiff, killdiff, is_blue):
    """
    minuto   : minuto atual do jogo (ex: 14)
    golddiff : gold diff do SEU time (positivo = na frente), lido da barra da transmissao
    killdiff : kills do seu time - kills do adversario
    is_blue  : 1 se seu time joga no lado azul, 0 se vermelho
    """
    cut = _nearest_cut(minuto)
    m = MODELS[cut]
    x = np.array([golddiff, killdiff, is_blue], dtype=float)
    x_scaled = (x - m['scaler_mean']) / m['scaler_scale']
    coef = np.array([m['coef']['golddiff'], m['coef']['killdiff'], m['coef']['is_blue']])
    z = np.dot(coef, x_scaled) + m['intercept']
    p = 1 / (1 + np.exp(-z))
    return float(p), cut


def edge_e_stake(p_modelo, preco_mercado, kelly_fracionado=0.4, teto_stake=0.05):
    """
    preco_mercado : preco do contrato na Polymarket (ex: 0.55 = 55 centavos)
    teto_stake    : trava manual de seguranca (fracao maxima do bankroll), default 5%
    """
    edge = p_modelo - preco_mercado
    kelly_full = max((p_modelo - preco_mercado) / (1 - preco_mercado), 0) if preco_mercado < 1 else 0
    stake = min(kelly_full * kelly_fracionado, teto_stake)
    return edge, stake


if __name__ == '__main__':
    # ============ SEUS DADOS AQUI (edite antes de rodar) ============
    minuto = 14              # minuto atual do jogo
    golddiff = 1800          # gold diff do seu time, lido da barra da transmissao
    killdiff = 2             # kills do seu time - kills do adversario
    is_blue = 1              # 1 = lado azul, 0 = lado vermelho
    preco_polymarket = 0.62  # preco atual do contrato na Polymarket
    # ===================================================================

    p, cut_usado = prob_vitoria(minuto, golddiff, killdiff, is_blue)
    edge, stake = edge_e_stake(p, preco_polymarket)

    print(f"Minuto {minuto} | golddiff={golddiff} | killdiff={killdiff} | lado={'Blue' if is_blue else 'Red'}")
    print(f"(modelo usado: corte de {cut_usado}min | AUC hist. {MODELS[cut_usado]['auc']:.3f})\n")
    print(f"Probabilidade do modelo:  {p*100:.1f}%")
    print(f"Preco da Polymarket:      {preco_polymarket*100:.1f}%")
    print(f"Edge:                     {edge*100:+.1f} pontos")
    print(f"Stake sugerido (teto 5%): {stake*100:.2f}% do bankroll")

    if edge <= 0.08:
        print("\n-> Edge abaixo do minimo recomendado (8 pontos). Sem entrada.")
