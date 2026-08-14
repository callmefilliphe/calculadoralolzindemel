import glob
import io
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

# Minutos que usaremos como marcos de treino
CUTS = [10, 15, 20, 25]
MODELS_OUTPUT = {}

print("Procurando arquivos CSV do Oracle's Elixir...")
files = glob.glob('*OraclesElixir*.csv')
dfs = []

for f in files:
    try:
        with open(f, 'rb') as fp:
            data = fp.read()
        # Tratamento para identificar início dos dados caso haja cabeçalho corrompido/binário
        idx = data.find(b'gameid,')
        if idx != -1:
            text = data[idx:].decode('utf-8', errors='replace')
            df_temp = pd.read_csv(io.StringIO(text), low_memory=False)
            dfs.append(df_temp)
            print(f"-> Carregado: {f} ({len(df_temp)} linhas)")
    except Exception as e:
        print(f"Erro ao processar {f}: {e}")

if not dfs:
    raise ValueError("Nenhum CSV do Oracle's Elixir encontrado ou válido!")

df = pd.concat(dfs, ignore_index=True)

# Filtra apenas as linhas consolidadas por time
df_team = df[df['position'] == 'team'].copy()
df_team['is_blue'] = (df_team['side'].astype(str).str.lower() == 'blue').astype(int)
df_team['result'] = pd.to_numeric(df_team['result'], errors='coerce')

print(f"Total de jogos de time consolidados: {len(df_team)}")

# Treinamento por corte de minuto
for cut in CUTS:
    g_col = f'golddiffat{cut}'
    k_col = f'killsat{cut}'
    d_col = f'deathsat{cut}'

    # Filtra partidas válidas que contenham os dados do minuto
    valid = df_team.dropna(subset=['result', g_col, k_col, d_col]).copy()
    valid['killdiff'] = valid[k_col] - valid[d_col]

    X = valid[[g_col, 'killdiff', 'is_blue']].values
    y = valid['result'].values

    # Padronização de escala
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Regressão Logística
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_scaled, y)

    # Avaliação do modelo
    preds_prob = clf.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, preds_prob)
    brier = brier_score_loss(y, preds_prob)

    MODELS_OUTPUT[cut] = {
        'coef': {
            'golddiff': float(clf.coef_[0][0]),
            'killdiff': float(clf.coef_[0][1]),
            'is_blue': float(clf.coef_[0][2])
        },
        'intercept': float(clf.intercept_[0]),
        'mean': [float(m) for m in scaler.mean_],
        'scale': [float(s) for s in scaler.scale_],
        'auc': round(float(auc), 3),
        'brier': round(float(brier), 4),
        'samples': int(len(valid))
    }
    print(f"Minuto {cut}m treinado com sucesso | Amostras: {len(valid)} | AUC: {auc:.3f}")

# Salva o arquivo JSON final
with open('models.json', 'w', encoding='utf-8') as out:
    json.dump(MODELS_OUTPUT, out, indent=2)

print("\nArquivo 'models.json' gerado com sucesso!")
