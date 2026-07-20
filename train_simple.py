import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
import warnings, pickle
warnings.filterwarnings('ignore')

FILES = [
    '/mnt/user-data/uploads/2024_LoL_esports_match_data_from_OraclesElixir.csv',
    '/mnt/user-data/uploads/2025_LoL_esports_match_data_from_OraclesElixir.csv',
    '/mnt/user-data/uploads/2026_LoL_esports_match_data_from_OraclesElixir.csv',
]
TIME_CUTS = [10, 15, 20, 25]
FEATURES = ['golddiff', 'killdiff', 'is_blue']

def load_team_rows():
    frames = []
    for f in FILES:
        df = pd.read_csv(f, low_memory=False)
        df = df[(df['position'] == 'team') & (df['datacompleteness'] == 'complete')]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

team_df = load_team_rows()
print(f"Jogos unicos: {team_df['gameid'].nunique()}")

results = {}
for t in TIME_CUTS:
    cols = [f'golddiffat{t}', f'killsat{t}', f'opp_killsat{t}']
    sub = team_df[['side', 'result', 'gamelength'] + cols].dropna()
    sub = sub[sub['gamelength'] > t * 60]
    sub['golddiff'] = sub[f'golddiffat{t}']
    sub['killdiff'] = sub[f'killsat{t}'] - sub[f'opp_killsat{t}']
    sub['is_blue'] = (sub['side'] == 'Blue').astype(int)

    X = sub[FEATURES].values
    y = sub['result'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)

    p_test = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)

    results[t] = {
        'coef': dict(zip(FEATURES, clf.coef_[0])),
        'intercept': clf.intercept_[0],
        'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
        'auc': auc, 'brier': brier, 'n': len(X),
    }
    print(f"t={t}min | n={len(X)} | AUC={auc:.4f} | Brier={brier:.4f}")

with open('/home/claude/models_simple.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nSalvo em models_simple.pkl")
